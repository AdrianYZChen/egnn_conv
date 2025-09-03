import unittest
from typing import List, Tuple

import torch
import torch.nn as nn

from egnn_clean import get_edges_batch
from unet3d import UNetHiddenOnly3D
from pharmcond import (
    PharmCondModel,
    sample_grid_at_nodes,
    splat_nodes_to_grid,
)


def make_linear_coord_grid(B: int, D: int, H: int, W: int, device) -> torch.Tensor:
    """
    Build a 3-channel volume whose values are exactly the voxel indices:
    ch0 = x_idx, ch1 = y_idx, ch2 = z_idx.  Shape: (B, 3, D, H, W)
    """
    xs = torch.arange(W, device=device).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    ys = torch.arange(H, device=device).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
    zs = torch.arange(D, device=device).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
    grid = torch.cat([xs, ys, zs], dim=1).to(torch.float32)
    return grid


def build_edges_variable(n_nodes_list: List[int]) -> Tuple[list, torch.Tensor]:
    """Fully connected edges per-graph for variable node counts."""
    rows, cols = [], []
    edge_attrs = []
    offset = 0
    for n in n_nodes_list:
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(offset + i)
                    cols.append(offset + j)
                    edge_attrs.append([1.0])
        offset += n
    edges = [torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    return edges, edge_attr


class TestPharmCondModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device("cuda"))

    def test_grid_to_node_sampling_linear_identity(self):
        """
        If a volume stores linear fields (x_idx, y_idx, z_idx),
        trilinear sampling at arbitrary continuous indices should reproduce those indices.
        """
        for device in self.devices:
            B, D, H, W = 1, 7, 6, 5
            grid = make_linear_coord_grid(B, D, H, W, device)

            # Coordinates metadata: idx -> world: world = min + idx * stride
            coordinates_min = torch.tensor([0.0, 0.0, 0.0], device=device)
            stride = torch.tensor([1.0, 1.0, 1.0], device=device)

            # Pick random index positions strictly inside the volume to avoid boundary artifacts
            N = 20
            x_idx = torch.rand(N, device=device) * (W - 1)
            y_idx = torch.rand(N, device=device) * (H - 1)
            z_idx = torch.rand(N, device=device) * (D - 1)
            idx_xyz = torch.stack([x_idx, y_idx, z_idx], dim=1)

            coords = coordinates_min + idx_xyz * stride  # to world coords
            node_batch = torch.zeros(N, dtype=torch.long, device=device)

            sampled = sample_grid_at_nodes(
                grid=grid,
                coords_xyz=coords,
                node_batch=node_batch,
                coordinates_min=coordinates_min,
                spatial_stride=stride,
                align_corners=True,
            )  # (N, 3)

            # ch0 ≈ x_idx, ch1 ≈ y_idx, ch2 ≈ z_idx
            self.assertTrue(torch.allclose(sampled[:, 0], x_idx, atol=1e-4, rtol=1e-4))
            self.assertTrue(torch.allclose(sampled[:, 1], y_idx, atol=1e-4, rtol=1e-4))
            self.assertTrue(torch.allclose(sampled[:, 2], z_idx, atol=1e-4, rtol=1e-4))

    def test_node_to_grid_splat_nearest_exact(self):
        """
        Nearest-neighbor splatting should place each node feature exactly into its nearest voxel.
        """
        for device in self.devices:
            B, C, D, H, W = 2, 4, 6, 5, 7
            coordinates_min = torch.tensor([0.0, 0.0, 0.0], device=device)
            stride = torch.tensor([1.0, 1.0, 1.0], device=device)

            # Build integer indices (exact voxel centers)
            nodes_per_batch = [6, 8]
            N = sum(nodes_per_batch)
            node_batch = torch.cat([
                torch.full((nodes_per_batch[0],), 0, dtype=torch.long, device=device),
                torch.full((nodes_per_batch[1],), 1, dtype=torch.long, device=device),
            ])

            # Space nodes across the grid
            xi = torch.randint(0, W, (N,), device=device)
            yi = torch.randint(0, H, (N,), device=device)
            zi = torch.randint(0, D, (N,), device=device)
            coords = coordinates_min + torch.stack([xi, yi, zi], dim=1).to(torch.float32) * stride

            node_feats = torch.randn(N, C, device=device)
            vol = splat_nodes_to_grid(
                node_feats=node_feats,
                coords_xyz=coords,
                node_batch=node_batch,
                grid_shape=(B, C, D, H, W),
                coordinates_min=coordinates_min,
                spatial_stride=stride,
                mode="nearest",
            )

            # Expected accumulation
            expected = torch.zeros(B, C, D, H, W, device=device)
            for n in range(N):
                b = node_batch[n].item()
                expected[b, :, zi[n], yi[n], xi[n]] += node_feats[n]

            self.assertTrue(torch.allclose(vol, expected, atol=1e-6, rtol=1e-6))

    def test_node_to_grid_splat_trilinear_conservation(self):
        """
        Trilinear splatting should conserve 'mass':
        sum over volume per-channel equals sum of node features per-channel (for nodes inside bounds).
        """
        for device in self.devices:
            B, C, D, H, W = 1, 3, 10, 9, 8
            coordinates_min = torch.tensor([0.0, 0.0, 0.0], device=device)
            stride = torch.tensor([1.0, 1.0, 1.0], device=device)

            # Put nodes away from boundaries to avoid clamping effects
            N = 25
            x_idx = 1.2 + torch.rand(N, device=device) * (W - 2.4)
            y_idx = 1.2 + torch.rand(N, device=device) * (H - 2.4)
            z_idx = 1.2 + torch.rand(N, device=device) * (D - 2.4)
            coords = coordinates_min + torch.stack([x_idx, y_idx, z_idx], dim=1) * stride
            node_batch = torch.zeros(N, dtype=torch.long, device=device)
            node_feats = torch.randn(N, C, device=device)

            vol = splat_nodes_to_grid(
                node_feats=node_feats,
                coords_xyz=coords,
                node_batch=node_batch,
                grid_shape=(B, C, D, H, W),
                coordinates_min=coordinates_min,
                spatial_stride=stride,
                mode="trilinear",
            )

            self.assertTrue(torch.allclose(
                vol.sum(dim=(0, 2, 3, 4)).squeeze(0),   # (C,)
                node_feats.sum(dim=0),
                atol=1e-5, rtol=1e-5
            ))

    def test_full_model_forward_backward(self):
        """
        End-to-end smoke test with gradients on CPU/GPU.
        """
        for device in self.devices:
            # Problem sizes
            batch_size = 2
            n_nodes_per_graph = 5
            N_total = batch_size * n_nodes_per_graph

            in_node_nf, out_node_nf = 6, 4
            in_grid_nf, out_grid_nf = 2, 1
            D = H = W = 12

            h_in = torch.randn(N_total, in_node_nf, device=device, requires_grad=True)
            # Coordinates inside the grid bounds
            coordinates_min = torch.tensor([-6.0, -6.0, -6.0], device=device)
            stride = torch.tensor([1.0, 1.0, 1.0], device=device)
            idx_xyz = torch.stack([
                torch.rand(N_total, device=device) * (W - 1),
                torch.rand(N_total, device=device) * (H - 1),
                torch.rand(N_total, device=device) * (D - 1),
            ], dim=1)
            x = (coordinates_min + idx_xyz * stride).clone().detach().requires_grad_(True)

            edges, edge_attr = get_edges_batch(n_nodes_per_graph, batch_size)
            edges = [e.to(device) for e in edges]
            edge_attr = edge_attr.to(device)

            grid_in = torch.randn(batch_size, in_grid_nf, D, H, W, device=device, requires_grad=True)

            model = PharmCondModel(
                in_node_nf=in_node_nf,
                out_node_nf=out_node_nf,
                in_grid_nf=in_grid_nf,
                out_grid_nf=out_grid_nf,
                node_hidden=64,
                grid_hidden=32,
                n_hybrid_blocks=2,
                edges_in_channels=edge_attr.shape[-1],
                splat_mode="trilinear",
            ).to(device)

            h_out, x_out, grid_out = model(
                h_in=h_in, x=x, edges=edges, grid_in=grid_in,
                coordinates_min=coordinates_min, spatial_stride=stride,
                n_nodes_per_graph=n_nodes_per_graph, edge_attr=edge_attr
            )

            self.assertEqual(h_out.shape, (N_total, out_node_nf))
            self.assertEqual(x_out.shape, (N_total, 3))
            self.assertEqual(grid_out.shape, (batch_size, out_grid_nf, D, H, W))

            # Make a scalar loss that touches all outputs (including coords)
            loss = h_out.sum() + grid_out.sum() + 0.1 * x_out.sum()
            loss.backward()

            # Gradients should exist
            self.assertIsNotNone(h_in.grad)
            self.assertIsNotNone(grid_in.grad)
            self.assertIsNotNone(x.grad)
            self.assertGreater(h_in.grad.abs().sum().item(), 0.0)
            self.assertGreater(grid_in.grad.abs().sum().item(), 0.0)
            self.assertGreater(x.grad.abs().sum().item(), 0.0)

            # Some parameter should receive gradients
            total_param_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_param_grad += float(p.grad.abs().sum().item())
            self.assertGreater(total_param_grad, 0.0)

    def test_model_with_unet_grid_and_variable_nodes(self):
        """
        Test the model using a UNet grid block and variable nodes per batch (via node_batch).
        """
        for device in self.devices:
            n_nodes_list = [3, 6]  # variable sizes for two graphs
            batch_size = len(n_nodes_list)
            N_total = sum(n_nodes_list)
            in_node_nf, out_node_nf = 5, 2
            in_grid_nf, out_grid_nf = 1, 1
            D = H = W = 10

            # Inputs
            h_in = torch.randn(N_total, in_node_nf, device=device)
            coordinates_min = torch.tensor([0.0, 0.0, 0.0], device=device)
            stride = torch.tensor([1.0, 1.0, 1.0], device=device)

            # Random coords inside volume
            idx_xyz = torch.stack([
                torch.rand(N_total, device=device) * (W - 1),
                torch.rand(N_total, device=device) * (H - 1),
                torch.rand(N_total, device=device) * (D - 1),
            ], dim=1)
            x = (coordinates_min + idx_xyz * stride)

            # Variable edges and node_batch
            edges, edge_attr = build_edges_variable(n_nodes_list)
            edges = [e.to(device) for e in edges]
            edge_attr = edge_attr.to(device)

            node_batch = torch.cat([
                torch.full((n_nodes_list[0],), 0, dtype=torch.long, device=device),
                torch.full((n_nodes_list[1],), 1, dtype=torch.long, device=device),
            ])

            grid_in = torch.randn(batch_size, in_grid_nf, D, H, W, device=device)

            # Small UNet-based grid block
            def grid_block_factory():
                return UNetHiddenOnly3D(
                    top_channels=32,             # matches grid_hidden below
                    num_lower_levels=1,
                    channels_per_lower_level=[48],
                    num_convs_per_level=2,
                    pooling_strides=2,
                )

            model = PharmCondModel(
                in_node_nf=in_node_nf,
                out_node_nf=out_node_nf,
                in_grid_nf=in_grid_nf,
                out_grid_nf=out_grid_nf,
                node_hidden=64,
                grid_hidden=32,
                n_hybrid_blocks=1,
                edges_in_channels=edge_attr.shape[-1],
                splat_mode="trilinear",
                grid_block_factory=grid_block_factory,
            ).to(device)

            h_out, x_out, grid_out = model(
                h_in=h_in, x=x, edges=edges, grid_in=grid_in,
                coordinates_min=coordinates_min, spatial_stride=stride,
                node_batch=node_batch, edge_attr=edge_attr
            )

            self.assertEqual(h_out.shape, (N_total, out_node_nf))
            self.assertEqual(x_out.shape, (N_total, 3))
            self.assertEqual(grid_out.shape, (batch_size, out_grid_nf, D, H, W))


if __name__ == "__main__":
    unittest.main(verbosity=2)
