from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .egnn_clean import E_GCL
    from .unet3d import BasicConv3D
except ImportError:
    from egnn_clean import E_GCL
    from unet3d import BasicConv3D


def _require_batch_index(
    n_nodes_total: int,
    node_batch: Optional[torch.Tensor],
    n_nodes_per_graph: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a (N_total,) tensor mapping every node to its batch id.
    If node_batch is None, infers it assuming fixed-size graphs with n_nodes_per_graph.
    """
    if node_batch is not None:
        return node_batch.to(device)
    if n_nodes_per_graph is None:
        raise ValueError("Provide node_batch or n_nodes_per_graph to map nodes to batch examples.")
    if n_nodes_total % n_nodes_per_graph != 0:
        raise ValueError("n_nodes_total must be divisible by n_nodes_per_graph.")
    batch_size = n_nodes_total // n_nodes_per_graph
    return torch.arange(batch_size, device=device).repeat_interleave(n_nodes_per_graph)


def _coords_to_continuous_grid_idx(
    coords_xyz: torch.Tensor,           # (N, 3) real-world coordinates (x, y, z)
    coordinates_min: torch.Tensor,      # (3,)
    spatial_stride: torch.Tensor,       # (3,)
) -> torch.Tensor:
    """
    Converts real-world coordinates to continuous grid indices in (x_idx, y_idx, z_idx).
    Index 0 corresponds to the center of the first voxel.
    """
    return (coords_xyz - coordinates_min) / spatial_stride  # (N, 3) in index units


def _grid_idx_to_normalized(
    idx_xyz: torch.Tensor,  # (N, 3) in index coordinates
    D: int, H: int, W: int,
) -> torch.Tensor:
    """
    Converts (x_idx, y_idx, z_idx) to normalized coordinates (x, y, z) in [-1, 1]
    expected by torch.nn.functional.grid_sample for 5D inputs.
    Note grid_sample expects order (x, y, z).
    """
    x = 2.0 * (idx_xyz[:, 0] / max(W - 1, 1)) - 1.0
    y = 2.0 * (idx_xyz[:, 1] / max(H - 1, 1)) - 1.0
    z = 2.0 * (idx_xyz[:, 2] / max(D - 1, 1)) - 1.0
    return torch.stack([x, y, z], dim=-1)  # (N, 3)


def sample_grid_at_nodes(
    grid: torch.Tensor,                 # (B, Cg, D, H, W)
    coords_xyz: torch.Tensor,           # (N, 3)
    node_batch: torch.Tensor,           # (N,)
    coordinates_min: torch.Tensor,      # (3,)
    spatial_stride: torch.Tensor,       # (3,)
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Samples grid features at node coordinates using trilinear interpolation
    (via grid_sample). Returns (N, Cg).
    """
    device = grid.device
    B, Cg, D, H, W = grid.shape
    N = coords_xyz.shape[0]

    # Continuous grid indices (x_idx, y_idx, z_idx)
    idx_xyz = _coords_to_continuous_grid_idx(coords_xyz, coordinates_min, spatial_stride)  # (N, 3)
    norm_xyz = _grid_idx_to_normalized(idx_xyz, D, H, W)  # (N, 3) in [-1, 1]

    # Pack per-batch grids of query points:
    # For each batch b, gather the nodes that belong to b.
    feats_per_batch = []
    for b in range(B):
        mask = (node_batch == b)
        if not torch.any(mask):
            # If a batch has no nodes, create an empty tensor on the right device
            feats_per_batch.append(torch.empty(0, Cg, device=device))
            continue
        nb = mask.sum().item()
        # grid_sample expects grid of shape (N, Dz, Hy, Wx, 3)
        # Use (Dz=nb, Hy=1, Wx=1) and reshape later.
        grid_b = norm_xyz[mask].view(1, nb, 1, 1, 3)  # (1, nb, 1, 1, 3)
        sampled = F.grid_sample(
            input=grid[b:b+1],  # (1, Cg, D, H, W)
            grid=grid_b,        # (1, nb, 1, 1, 3)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners,
        )  # -> (1, Cg, nb, 1, 1)
        feats_per_batch.append(sampled.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0))  # (nb, Cg)

    # Concatenate back in original node order
    result = torch.empty(N, Cg, device=device)
    for b in range(B):
        mask = (node_batch == b)
        if torch.any(mask):
            result[mask] = feats_per_batch[b]
    return result  # (N, Cg)


def splat_nodes_to_grid(
    node_feats: torch.Tensor,           # (N, Cg_out)
    coords_xyz: torch.Tensor,           # (N, 3)
    node_batch: torch.Tensor,           # (N,)
    grid_shape: Tuple[int, int, int, int, int],  # (B, Cg_out, D, H, W)
    coordinates_min: torch.Tensor,      # (3,)
    spatial_stride: torch.Tensor,       # (3,)
    mode: Literal["nearest", "trilinear"] = "trilinear",
) -> torch.Tensor:
    """
    Projects node features into a 3D volume by (optionally) trilinear splatting.

    Returns:
        volume: (B, Cg_out, D, H, W)
    """
    device = node_feats.device
    B, Cg_out, D, H, W = grid_shape

    vol = torch.zeros((B, Cg_out, D, H, W), device=device)

    # Continuous index coordinates (x_idx, y_idx, z_idx)
    idx = _coords_to_continuous_grid_idx(coords_xyz, coordinates_min, spatial_stride)  # (N, 3)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]

    vol_flat = vol.view(B, Cg_out, -1)

    if mode == "nearest":
        xi_raw = torch.round(x).long()
        yi_raw = torch.round(y).long()
        zi_raw = torch.round(z).long()

        valid = (
            (xi_raw >= 0) & (xi_raw < W) &
            (yi_raw >= 0) & (yi_raw < H) &
            (zi_raw >= 0) & (zi_raw < D)
        )

        # Clamp for flat index computation, but only add valid nodes
        xi = xi_raw.clamp(0, W - 1)
        yi = yi_raw.clamp(0, H - 1)
        zi = zi_raw.clamp(0, D - 1)

        flat_index = zi * (H * W) + yi * W + xi  # (N,)
        for b in range(B):
            mask_b = (node_batch == b) & valid
            if torch.any(mask_b):
                flat_b = flat_index[mask_b]                    # (N_b,)
                src = node_feats[mask_b].transpose(0, 1)       # (C, N_b)
                idx = flat_b.unsqueeze(0).expand(Cg_out, -1)   # (C, N_b)
                vol_flat[b].scatter_add_(1, idx, src)
        return vol

    # Trilinear splatting
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()
    x1 = (x0 + 1)
    y1 = (y0 + 1)
    z1 = (z0 + 1)

    # Fractions
    fx = (x - x0.float()).clamp(0, 1)
    fy = (y - y0.float()).clamp(0, 1)
    fz = (z - z0.float()).clamp(0, 1)

    # In-bounds check to avoid duplicating weights at borders
    def in_bounds(a, b, c):
        return (
            (a >= 0) & (a < W) &
            (b >= 0) & (b < H) &
            (c >= 0) & (c < D)
        )

    corners = [
        (x0, y0, z0, (1 - fx) * (1 - fy) * (1 - fz)),
        (x1, y0, z0, (fx)     * (1 - fy) * (1 - fz)),
        (x0, y1, z0, (1 - fx) * (fy)     * (1 - fz)),
        (x1, y1, z0, (fx)     * (fy)     * (1 - fz)),
        (x0, y0, z1, (1 - fx) * (1 - fy) * (fz)),
        (x1, y0, z1, (fx)     * (1 - fy) * (fz)),
        (x0, y1, z1, (1 - fx) * (fy)     * (fz)),
        (x1, y1, z1, (fx)     * (fy)     * (fz)),
    ]

    vol_flat = vol.view(B, Cg_out, -1)
    for xx, yy, zz, w in corners:
        valid = in_bounds(xx, yy, zz)
        # Clamp for index computation, but mask invalid weights to avoid border duplication
        xi = xx.clamp(0, W - 1)
        yi = yy.clamp(0, H - 1)
        zi = zz.clamp(0, D - 1)
        flat = zi * (H * W) + yi * W + xi  # (N,)
        w_valid = w * valid.float()
        contrib = node_feats * w_valid.unsqueeze(1)            # (N, C)

        for b in range(B):
            mask_b = (node_batch == b) & valid
            if torch.any(mask_b):
                flat_b = flat[mask_b]                          # (N_b,)
                src = contrib[mask_b].transpose(0, 1)          # (C, N_b)
                idx = flat_b.unsqueeze(0).expand(Cg_out, -1)   # (C, N_b)
                vol_flat[b].scatter_add_(1, idx, src)
    return vol


class NodeGridInteraction(nn.Module):
    """
    Bi-directional interaction between graph nodes and 3D grid.

    - Grid -> Node: sample grid features at node coords and update node features with MLP.
    - Node -> Grid: project node features into a grid volume (splat) and update grid with 1x1x1 conv.

    Both paths use residual updates.
    """
    def __init__(
        self,
        node_channels: int,
        grid_channels: int,
        hidden: int = 128,
        splat_mode: Literal["nearest", "trilinear"] = "trilinear",
    ):
        super().__init__()
        self.splat_mode = splat_mode

        # Node update (concat node + sampled grid -> node)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_channels + grid_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, node_channels),
        )
        self.node_norm = nn.LayerNorm(node_channels)

        # Node -> Grid projection
        self.node_to_grid = nn.Linear(node_channels, grid_channels)

        # Grid update (concat grid + splatted nodes -> grid)
        self.grid_mlp = nn.Sequential(
            nn.Conv3d(grid_channels * 2, grid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(grid_channels, grid_channels, kernel_size=1),
        )
        self.grid_norm = nn.InstanceNorm3d(grid_channels, affine=True)

    def forward(
        self,
        h: torch.Tensor,                    # (N, Cn)
        x: torch.Tensor,                    # (N, 3) node coordinates (x, y, z)
        grid: torch.Tensor,                 # (B, Cg, D, H, W)
        node_batch: torch.Tensor,           # (N,)
        coordinates_min: torch.Tensor,      # (3,)
        spatial_stride: torch.Tensor,       # (3,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Cg, D, H, W = grid.shape

        # 1) Grid -> Node
        sampled = sample_grid_at_nodes(
            grid=grid,
            coords_xyz=x,
            node_batch=node_batch,
            coordinates_min=coordinates_min,
            spatial_stride=spatial_stride,
            align_corners=True,
        )  # (N, Cg)
        h = self.node_norm(h + self.node_mlp(torch.cat([h, sampled], dim=-1)))  # residual

        # 2) Node -> Grid
        node_proj = self.node_to_grid(h)  # (N, Cg)
        splat = splat_nodes_to_grid(
            node_feats=node_proj,
            coords_xyz=x,
            node_batch=node_batch,
            grid_shape=grid.shape,
            coordinates_min=coordinates_min,
            spatial_stride=spatial_stride,
            mode=self.splat_mode,
        )  # (B, Cg, D, H, W)

        grid = self.grid_norm(grid + self.grid_mlp(torch.cat([grid, splat], dim=1)))  # residual
        return h, grid


class HybridBlock(nn.Module):
    def __init__(
        self,
        node_channels: int,
        grid_channels: int,
        edges_in_channels: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        coord_normalize: bool = False,
        splat_mode: Literal["nearest", "trilinear"] = "trilinear",
        grid_block_factory=None,
    ):
        super().__init__()

        # If caller doesn't provide a grid block, use a simple 2x Conv3d block
        if grid_block_factory is None:
            def grid_block_factory():
                return BasicConv3D(in_channels=grid_channels, out_channels=grid_channels, num_convs=2)

        # example using unet

        # from unet3d import UNetHiddenOnly3D
        # def grid_block_factory():
        #     return UNetHiddenOnly3D(top_channels=64, num_lower_levels=2, channels_per_lower_level=[96, 128])


        # Top path
        self.gnn1 = E_GCL(
            input_nf=node_channels,
            output_nf=node_channels,
            hidden_nf=node_channels,
            edges_in_d=edges_in_channels,
            act_fn=act_fn,
            residual=True,
            attention=False,
            normalize=coord_normalize,
            tanh=False,
        )
        self.grid1 = grid_block_factory()

        # Interaction
        self.interact = NodeGridInteraction(
            node_channels=node_channels,
            grid_channels=grid_channels,
            hidden=max(node_channels, grid_channels),
            splat_mode=splat_mode,
        )

        # Bottom path
        self.gnn2 = E_GCL(
            input_nf=node_channels,
            output_nf=node_channels,
            hidden_nf=node_channels,
            edges_in_d=edges_in_channels,
            act_fn=act_fn,
            residual=True,
            attention=False,
            normalize=coord_normalize,
            tanh=False,
        )
        self.grid2 = grid_block_factory()

    def forward(
        self,
        h: torch.Tensor,                    # (N, Cn)
        x: torch.Tensor,                    # (N, 3)
        edges: torch.Tensor,                # (2, E)
        grid: torch.Tensor,                 # (B, Cg, D, H, W)
        node_batch: torch.Tensor,           # (N,)
        coordinates_min: torch.Tensor,      # (3,)
        spatial_stride: torch.Tensor,       # (3,)
        edge_attr: Optional[torch.Tensor] = None,
        node_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Top path
        h, x, _ = self.gnn1(h, edges, x, edge_attr=edge_attr, node_attr=node_attr)
        grid = self.grid1(grid)

        # Interaction
        h, grid = self.interact(
            h=h, x=x, grid=grid, node_batch=node_batch,
            coordinates_min=coordinates_min, spatial_stride=spatial_stride
        )

        # Bottom path
        h, x, _ = self.gnn2(h, edges, x, edge_attr=edge_attr, node_attr=node_attr)
        grid = self.grid2(grid)

        return h, x, grid


class PharmCondModel(nn.Module):
    """
    Full model that stacks multiple HybridBlocks.
    Includes simple input/output heads for nodes and grid.

    Inputs:
        - h_in: (N, in_node_nf)
        - x:    (N, 3)
        - edges: (2, E)
        - grid_in: (B, in_grid_nf, D, H, W)
        - node_batch: (N,)
        - coordinates_min: (3,)
        - spatial_stride: (3,)

    Returns:
        h_out: (N, out_node_nf)
        x:     (N, 3)
        grid_out: (B, out_grid_nf, D, H, W)
    """
    def __init__(
        self,
        in_node_nf: int,
        out_node_nf: int,
        in_grid_nf: int,
        out_grid_nf: int,
        node_hidden: int = 128,
        grid_hidden: int = 64,
        n_hybrid_blocks: int = 2,
        edges_in_channels: int = 0,
        splat_mode: Literal["nearest", "trilinear"] = "trilinear",
        grid_block_factory=None,
    ):
        super().__init__()
        self.node_in = nn.Linear(in_node_nf, node_hidden)
        self.grid_in = nn.Conv3d(in_grid_nf, grid_hidden, kernel_size=1)

        self.blocks = nn.ModuleList([
            HybridBlock(
                node_channels=node_hidden,
                grid_channels=grid_hidden,
                edges_in_channels=edges_in_channels,
                splat_mode=splat_mode,
                grid_block_factory=grid_block_factory,
            ) for _ in range(n_hybrid_blocks)
        ])

        self.node_out = nn.Linear(node_hidden, out_node_nf)
        self.grid_out = nn.Conv3d(grid_hidden, out_grid_nf, kernel_size=1)

    def forward(
        self,
        h_in: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor,
        grid_in: torch.Tensor,
        coordinates_min: torch.Tensor,
        spatial_stride: torch.Tensor,
        node_batch: Optional[torch.Tensor] = None,
        n_nodes_per_graph: Optional[int] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_attr: Optional[torch.Tensor] = None,
    ):
        device = h_in.device
        N = h_in.shape[0]

        # Map nodes to batches
        node_batch = _require_batch_index(
            n_nodes_total=N,
            node_batch=node_batch,
            n_nodes_per_graph=n_nodes_per_graph,
            device=device,
        )

        # Input heads
        h = self.node_in(h_in)
        grid = self.grid_in(grid_in)

        # Hybrid stack
        for blk in self.blocks:
            h, x, grid = blk(
                h=h, x=x, edges=edges, grid=grid, node_batch=node_batch,
                coordinates_min=coordinates_min, spatial_stride=spatial_stride,
                edge_attr=edge_attr, node_attr=node_attr
            )

        # Output heads
        h_out = self.node_out(h)
        grid_out = self.grid_out(grid)
        return h_out, x, grid_out


if __name__ == "__main__":
    import torch
    try:
        from .egnn_clean import get_edges_batch
    except ImportError:
        from egnn_clean import get_edges_batch

    # Dummy parameters
    batch_size = 2
    n_nodes_per_graph = 4
    in_node_nf = 6
    out_node_nf = 3
    in_grid_nf = 2
    out_grid_nf = 1

    D = H = W = 32  # grid size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inputs
    N_total = batch_size * n_nodes_per_graph
    h_in = torch.randn(N_total, in_node_nf, device=device)
    x = torch.randn(N_total, 3, device=device)  # real-world coords (x, y, z), consistent with your dataset

    # Fully connected edges per graph (as in your example)
    edges, edge_attr = get_edges_batch(n_nodes_per_graph, batch_size)
    edges = torch.stack(edges).to(device)               # (2, E)
    edge_attr = edge_attr.to(device)                    # (E, 1)

    # 3D grid input
    grid_in = torch.randn(batch_size, in_grid_nf, D, H, W, device=device)

    # Grid metadata (first voxel center and stride between voxels)
    coordinates_min = torch.tensor([ -8.0, -8.0, -8.0 ], device=device)  # (x0, y0, z0)
    spatial_stride   = torch.tensor([  0.5,  0.5,  0.5 ], device=device)  # spacing

    # Model
    model = PharmCondModel(
        in_node_nf=in_node_nf,
        out_node_nf=out_node_nf,
        in_grid_nf=in_grid_nf,
        out_grid_nf=out_grid_nf,
        node_hidden=128,
        grid_hidden=64,
        n_hybrid_blocks=2,
        edges_in_channels=edge_attr.shape[-1],
        splat_mode="trilinear",
    ).to(device)

    h_out, x_out, grid_out = model(
        h_in=h_in,
        x=x,
        edges=edges,
        grid_in=grid_in,
        coordinates_min=coordinates_min,
        spatial_stride=spatial_stride,
        n_nodes_per_graph=n_nodes_per_graph,
        edge_attr=edge_attr,
    )

    assert h_out.shape == (N_total, out_node_nf)
    assert x_out.shape == (N_total, 3)
    assert grid_out.shape == (batch_size, out_grid_nf, D, H, W)
