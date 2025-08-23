import torch

class SpatialGridTensor3D():
    """
    Wrapper for a 3D spatial grid tensor that supports convertion from index to coordinates and vice versa.
    
    Inputs:
        grid_tensor: torch.Tensor, assuming last three dimensions are the grid dimensions
        coordinates_min: torch.Tensor, shape (3,), specifies coordinates of the first grid cell
        spatial_stride: torch.Tensor, shape (3,), specifies the stride between grid cells
    """
    def __init__(
        self, 
        grid_tensor: torch.Tensor, 
        coordinates_min: torch.Tensor,
        spatial_stride: torch.Tensor,
    ):
        assert grid_tensor.ndim > 2, "Grid tensor must have at least 3 dimensions"
        assert coordinates_min.shape == (3,), "Coordinates min must be a 1D tensor of shape (3,)"
        assert spatial_stride.shape == (3,), "Spatial stride must be a 1D tensor of shape (3,)"
        
        self.grid_tensor = grid_tensor
        self.coordinates_min = coordinates_min
        self.spatial_stride = spatial_stride

    def index_to_coordinates(self, index: torch.Tensor) -> torch.Tensor:
        return self.coordinates_min + index * self.spatial_stride

    def coordinates_to_index(self, coordinates: torch.Tensor) -> torch.Tensor:
        return (coordinates - self.coordinates_min) // self.spatial_stride