"""
Localization Utilities Package
"""

from .geometry import (
    wrap_angle,
    world_to_grid,
    grid_to_world,
    rotation_matrix,
    transform_scan_to_world
)

from .map_loader import (
    load_map,
    MapInfo
)

from .data_format import (
    load_sensor_data,
    save_estimates,
    save_timing
)

from .sparse_loader import load_sparse_sensor_data

from .noise_models import SensorNoiseModel

from .config_loader import load_config

from .distance_transform import compute_distance_transform

from .likelihood_field import compute_likelihood_field

__version__ = "1.0.0"
__all__ = [
    # Geometry
    'wrap_angle',
    'world_to_grid',
    'grid_to_world',
    'rotation_matrix',
    'transform_scan_to_world',
    # Map
    'load_map',
    'MapInfo',
    # Data
    'load_sensor_data',
    'save_estimates',
    'save_timing',
    # Noise
    'SensorNoiseModel',
    # Config
    'load_config',
    # Algorithm-specific
    'compute_distance_transform',
    'compute_likelihood_field',
]
