import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable


# Default bounds for different parameter types
DEFAULT_BOUNDS = {
    'loss': (0.0, 1e-4),
    'reflectivity': (0.0, 1.0),
    'tuning': (-180.0, 180.0),
    'alpha': (0.0, 90.0),
    'mass': (1.0, 100.0),
    'power': (0.0, 10.0),
    'phase': (-180.0, 180.0),
    'db': (0.0, 20.0),
    'angle': (-180.0, 180.0),
    'length': (0.1, 4000.0),
}

# Relative distance bounds (fixed for all relative distances)
RELATIVE_DISTANCE_BOUNDS = (0.01, 0.49)


def prepare_transform_fn(
    params: list[tuple[str, str]], 
    n: int, 
    bounds: dict[str, tuple[float, float]] | None = None
) -> tuple[Callable, list[str]]:
    """
    Prepare a transformation function that converts unbounded optimization parameters 
    to bounded physical parameters.
    
    This function analyzes the params list from sparse_uifo to determine:
    1. Which parameters are physical (direct optimization)
    2. Which parameters are lengths (computed from positional encoding)
    
    It returns a JIT-compilable, differentiable function that transforms
    unbounded normally-distributed scaled parameters to bounded physical parameters.
    
    Parameters
    ----------
    params : list[tuple[str, str]]
        List of (element_name, property_name) tuples from sparse_uifo's output.
        E.g., [('m01', 'loss'), ('m01', 'reflectivity'), ..., ('boundary01_center11', 'length'), ...]
    n : int
        Grid size (n x n grid)
    bounds : dict[str, tuple[float, float]] | None
        Optional dictionary mapping property names to (lower, upper) bounds.
        If None, DEFAULT_BOUNDS are used. Can override specific bounds.
        E.g., {'mass': (0, 100), 'length': (0, 4000)}
        
    Returns
    -------
    restore_params_fn : Callable
        Function with signature: scaled_params (M,) -> params (N,)
        Where M is the number of optimization parameters and N is len(params).
        Can be vmapped for batch processing: (B, M) -> (B, N)
    optimization_param_names : list[str]
        List of parameter names for the optimization parameters.
        E.g., ['m01_loss', 'm01_reflectivity', ..., 'boundary_dist_01', 'row_spacing_01', ...]
        
    Examples
    --------
    >>> setup, params = sparse_uifo(element_array, n)
    >>> restore_params_fn, opt_names = prepare_transform_fn(params, n, bounds={'mass': (0, 100)})
    >>> scaled_params = jax.random.normal(key, (len(opt_names),))
    >>> physical_params = restore_params_fn(scaled_params)
    >>> # For batch processing:
    >>> restore_batch = jax.vmap(restore_params_fn)
    >>> batch_params = restore_batch(batch_scaled_params)  # (B, M) -> (B, N)
    """
    # Merge default bounds with user-provided bounds
    effective_bounds = DEFAULT_BOUNDS.copy()
    if bounds is not None:
        effective_bounds.update(bounds)
    
    # Separate physical parameters from length parameters and fixed parameters
    physical_param_indices = []  # Index in output params
    physical_param_names = []    # Name for optimization
    physical_param_bounds = []   # (lower, upper) bounds
    
    length_param_indices = []    # Index in output params
    length_connection_names = [] # Connection name for length computation
    
    fixed_param_indices = []     # Index in output params for fixed parameters
    fixed_param_values = []      # Fixed values (e.g., refractive_index = 1.0)
    
    for i, (element_name, property_name) in enumerate(params):
        if property_name == 'length':
            # Length parameter - computed from positional encoding
            length_param_indices.append(i)
            length_connection_names.append(element_name)  # element_name is actually connection_name for lengths
        elif property_name == 'refractive_index':
            # Fixed parameter - always 1.0
            fixed_param_indices.append(i)
            fixed_param_values.append(1.0)
        else:
            # Physical parameter - direct optimization
            physical_param_indices.append(i)
            physical_param_names.append(f"{element_name}_{property_name}")
            physical_param_bounds.append(effective_bounds.get(property_name, (0.0, 1.0)))
    
    # Get positional encoding parameter names
    positional_encoding_names = get_positional_encoding_names(n)
    
    # Build batch indices for length computation
    assert len(length_connection_names) > 0, "No length parameters found in params"
    all_length_indices = [get_length_indices(conn, n) for conn in length_connection_names]
    batch_length_indices = prepare_batch_indices(all_length_indices)
    
    # Build optimization parameter names
    # Order: physical parameters first, then positional encoding
    optimization_param_names = physical_param_names + list(positional_encoding_names)
    
    # Number of parameters
    n_physical = len(physical_param_indices)
    n_positional = len(positional_encoding_names)
    n_output = len(params)
    
    # Pre-compute arrays for efficient transformation
    physical_lower = jnp.array([b[0] for b in physical_param_bounds], dtype=jnp.float32)
    physical_upper = jnp.array([b[1] for b in physical_param_bounds], dtype=jnp.float32)
    physical_range = physical_upper - physical_lower
    
    physical_output_idx = jnp.array(physical_param_indices, dtype=jnp.int32)
    length_output_idx = jnp.array(length_param_indices, dtype=jnp.int32)
    
    # Fixed parameters (e.g., refractive_index = 1.0)
    fixed_output_idx = jnp.array(fixed_param_indices, dtype=jnp.int32) if fixed_param_indices else None
    fixed_values = jnp.array(fixed_param_values, dtype=jnp.float32) if fixed_param_values else None
    
    # Positional encoding structure:
    # [boundary_dist(4n), row_spacing(n-1), col_spacing(n-1), boundary_mirror_rel(4n), cell_mirror_rel(4n²)]
    n_boundary_dist = 4 * n
    n_row_spacing = n - 1
    n_col_spacing = n - 1
    n_boundary_mirror_rel = 4 * n
    n_cell_mirror_rel = 4 * n * n
    
    # Indices for different sections of positional encoding
    boundary_dist_start = 0
    boundary_dist_end = n_boundary_dist
    row_spacing_start = boundary_dist_end
    row_spacing_end = row_spacing_start + n_row_spacing
    col_spacing_start = row_spacing_end
    col_spacing_end = col_spacing_start + n_col_spacing
    boundary_mirror_rel_start = col_spacing_end
    boundary_mirror_rel_end = boundary_mirror_rel_start + n_boundary_mirror_rel
    cell_mirror_rel_start = boundary_mirror_rel_end
    cell_mirror_rel_end = cell_mirror_rel_start + n_cell_mirror_rel
    
    # Length bounds - convert to float32 for consistent types
    length_bounds = effective_bounds.get('length', (0.1, 4000.0))
    length_lower = jnp.float32(length_bounds[0])
    length_upper = jnp.float32(length_bounds[1])
    length_range = length_upper - length_lower
    
    rel_lower = jnp.float32(RELATIVE_DISTANCE_BOUNDS[0])
    rel_upper = jnp.float32(RELATIVE_DISTANCE_BOUNDS[1])
    rel_range = rel_upper - rel_lower
    
    def restore_params_fn(scaled_params: jnp.ndarray) -> jnp.ndarray:
        """
        Transform unbounded scaled parameters to bounded physical parameters.
        
        Parameters
        ----------
        scaled_params : jnp.ndarray
            Unbounded optimization parameters, shape (M,)
            
        Returns
        -------
        jnp.ndarray
            Bounded physical parameters, shape (N,) matching params order
        """
        # Ensure input is float32 for consistent types
        scaled_params = scaled_params.astype(jnp.float32)
        
        # Initialize output array
        output = jnp.zeros(n_output, dtype=jnp.float32)
        
        # Split scaled_params into physical and positional sections
        scaled_physical = scaled_params[:n_physical]
        scaled_positional = scaled_params[n_physical:n_physical + n_positional]
        
        # Transform physical parameters: sigmoid(x) * range + lower
        # This maps N(0,1) → approximately Uniform(lower, upper)
        physical_transformed = jax.nn.sigmoid(scaled_physical) * physical_range + physical_lower
        
        # Place physical parameters at their output indices
        output = output.at[physical_output_idx].set(physical_transformed)
        
        # Transform positional encoding to distance_array
        # Different transformations for different sections:
        # - Boundary distances: absolute lengths (length bounds)
        # - Row/col spacing: absolute lengths (length bounds)
        # - Relative distances: relative to base (RELATIVE_DISTANCE_BOUNDS)
        
        # Boundary source distances (absolute)
        boundary_dist = jax.nn.sigmoid(scaled_positional[boundary_dist_start:boundary_dist_end])
        boundary_dist = boundary_dist * length_range + length_lower
        
        # Row spacing (absolute)
        row_spacing = jax.nn.sigmoid(scaled_positional[row_spacing_start:row_spacing_end])
        row_spacing = row_spacing * length_range + length_lower
        
        # Col spacing (absolute)
        col_spacing = jax.nn.sigmoid(scaled_positional[col_spacing_start:col_spacing_end])
        col_spacing = col_spacing * length_range + length_lower
        
        # Boundary mirror relative distances (relative)
        boundary_mirror_rel = jax.nn.sigmoid(scaled_positional[boundary_mirror_rel_start:boundary_mirror_rel_end])
        boundary_mirror_rel = boundary_mirror_rel * rel_range + rel_lower
        
        # Cell mirror relative distances (relative)
        cell_mirror_rel = jax.nn.sigmoid(scaled_positional[cell_mirror_rel_start:cell_mirror_rel_end])
        cell_mirror_rel = cell_mirror_rel * rel_range + rel_lower
        
        # Assemble distance_array
        distance_array = jnp.concatenate([
            boundary_dist,
            row_spacing,
            col_spacing,
            boundary_mirror_rel,
            cell_mirror_rel
        ])
        
        # Compute lengths if there are any length parameters
        lengths = compute_lengths_batch(distance_array, batch_length_indices)
        output = output.at[length_output_idx].set(lengths)
        
        # Set fixed parameters (e.g., refractive_index = 1.0)
        if fixed_output_idx is not None:
            output = output.at[fixed_output_idx].set(fixed_values)
        
        return output
    
    return restore_params_fn, optimization_param_names


def get_active_parameters_mask(element_array: jnp.ndarray, sparse_setup: dict, n: int) -> jnp.ndarray:
    """
    Generate a mask of active parameters for a given element array and sparse setup.

    The parameter order is as follows:
    - boundary source parameters (2 parameters per source x n x 4 in total; clockwise starting from top left)
    For detector, the parameters are ignored. For laser, the parameters are the power and phase. For squeezer, the parameters are the db and angle.
    - boundary mirror parameters (4 parameters per mirror x n x 4 in total; clockwise starting from top left)
    For mirror, the parameters are the loss, reflectivity, tuning, and mass.
    - cell mirror parameters (4 parameters per mirror x n x n x 4 in total; cell first (clockwise starting from left), then rows (from left to right), then columns (from top to bottom))
    For mirror, the parameters are the loss, reflectivity, tuning, and mass.
    - cell beamsplitter parameters (5 parameters per beamsplitter x n x n in total; cell first (clockwise starting from top left), then rows (from left to right), then columns (from top to bottom))
    For beamsplitter, the parameters are the loss, reflectivity, tuning, alpha, and mass.
    For directional beamsplitter, the parameters are ignored completely.

    The physical parameters are followed by the positional encoding parameters:
    - n*4 boundary source distances (clockwise starting from top left)
    - (n-1) row spacing distances (distances between consecutive rows)
    - (n-1) col spacing distances (distances between consecutive columns)
    - n*4 boundary mirror relative distances (clockwise starting from top left)
    - n*n*4 cell mirror relative distances (top-right-bottom-left, then by row, then by col)

    Therefore, the total number of parameters is:
    N_parameters = 25 * n ** 2 + 34 * n - 2.

    Parameters
    ----------
    element_array : jnp.ndarray
        The element array. Shape: (N_elements,) where N_elements = n * (8 + 5*n).
    sparse_setup : dict
        The sparse setup.
    n : int
        Grid size (n x n grid)

    Returns
    -------
    jnp.ndarray
        The boolean mask of active parameters. Shape: (N_parameters,)
    """

    # We don't need this function to be too efficient since it is only used once per setup.
    # So we use a simple and readable implementation.

    # Total parameter count calculation:
    # Physical: 8n (boundary sources) + 16n (boundary mirrors) + 16n² (cell mirrors) + 5n² (cell beamsplitters)
    # Positional: 4n (boundary source distances) + (n-1) (row spacing) + (n-1) (col spacing) + 4n (boundary mirror rel) + 4n² (cell mirror rel)
    # Total: 21n² + 8n + 16n + 4n + 2(n-1) = 21n² + 28n + 2n - 2 = ... wait, let me recalculate
    # Physical: 2*4n + 4*4n + 4*4n² + 5n² = 8n + 16n + 16n² + 5n² = 21n² + 24n
    # Positional: 4n + (n-1) + (n-1) + 4n + 4n² = 8n + 2n - 2 + 4n² = 4n² + 10n - 2
    # Total: 21n² + 24n + 4n² + 10n - 2 = 25n² + 34n - 2 ✓
    N_parameters = 25 * n ** 2 + 34 * n - 2
    
    # Initialize all parameters as inactive
    mask = jnp.zeros(N_parameters, dtype=bool)
    
    # Extract element subsets from the flat element_array
    # Array structure: [boundary_sources(4n), boundary_mirrors(4n), cell_mirrors(4n²), cell_beamsplitters(n²)]
    n_boundary_sources = 4 * n
    n_boundary_mirrors = 4 * n
    n_cell_mirrors = 4 * n * n
    n_cell_beamsplitters = n * n
    
    # Split the element array into sections
    boundary_sources = element_array[:n_boundary_sources]
    boundary_mirrors = element_array[n_boundary_sources:n_boundary_sources + n_boundary_mirrors]
    cell_mirrors = element_array[n_boundary_sources + n_boundary_mirrors:n_boundary_sources + n_boundary_mirrors + n_cell_mirrors]
    cell_beamsplitters = element_array[n_boundary_sources + n_boundary_mirrors + n_cell_mirrors:]
    
    # Track the current position in the parameter array
    param_idx = 0
    
    # ===== PHYSICAL PARAMETERS =====
    
    # 1. BOUNDARY SOURCE PARAMETERS (2 params per source × 4n sources = 8n params)
    # Detector (1): no parameters
    # Laser (2): power, phase
    # Squeezer (3): db, angle
    for i in range(n_boundary_sources):
        element_type = boundary_sources[i]
        
        # Check if this is a laser or squeezer (both have 2 parameters)
        is_laser = (element_type == 2)
        is_squeezer = (element_type == 3)
        
        if is_laser or is_squeezer:
            # Activate both parameters for laser/squeezer
            mask = mask.at[param_idx:param_idx + 2].set(True)
        
        # Move to next source's parameter slots (2 params per source)
        param_idx += 2
    
    # 2. BOUNDARY MIRROR PARAMETERS (4 params per mirror × 4n mirrors = 16n params)
    # Element types: 0 (nothing), 4 (mirror), 5 (mirror with free mass)
    # Parameters: loss, reflectivity, tuning, mass
    for i in range(n_boundary_mirrors):
        element_type = boundary_mirrors[i]
        
        # Check if this is a mirror (with or without free mass)
        is_mirror = (element_type == 4)
        is_mirror_with_mass = (element_type == 5)
        
        if is_mirror:
            # Activate first 3 parameters (loss, reflectivity, tuning), but not mass
            mask = mask.at[param_idx:param_idx + 3].set(True)
        elif is_mirror_with_mass:
            # Activate all 4 parameters (loss, reflectivity, tuning, mass)
            mask = mask.at[param_idx:param_idx + 4].set(True)
        
        # Move to next mirror's parameter slots (4 params per mirror)
        param_idx += 4
    
    # 3. CELL MIRROR PARAMETERS (4 params per mirror × 4n² mirrors = 16n² params)
    # Same structure as boundary mirrors
    # Order: cell first (clockwise: left, top, right, bottom), then by row (left to right), then by col (top to bottom)
    for i in range(n_cell_mirrors):
        element_type = cell_mirrors[i]
        
        is_mirror = (element_type == 4)
        is_mirror_with_mass = (element_type == 5)
        
        if is_mirror:
            # Activate first 3 parameters (loss, reflectivity, tuning)
            mask = mask.at[param_idx:param_idx + 3].set(True)
        elif is_mirror_with_mass:
            # Activate all 4 parameters (loss, reflectivity, tuning, mass)
            mask = mask.at[param_idx:param_idx + 4].set(True)
        
        # Move to next mirror's parameter slots (4 params per mirror)
        param_idx += 4
    
    # 4. CELL BEAMSPLITTER PARAMETERS (5 params per beamsplitter × n² beamsplitters = 5n² params)
    # Beamsplitter types: 6-9 (regular beamsplitter, possibly with mass)
    # Directional beamsplitter types: 10-17 (no parameters)
    # Parameters: loss, reflectivity, tuning, alpha, mass
    for i in range(n_cell_beamsplitters):
        element_type = cell_beamsplitters[i]
        
        # Regular beamsplitters: 6-7 (without mass), 8-9 (with mass)
        is_beamsplitter_no_mass = (element_type == 6) | (element_type == 7)
        is_beamsplitter_with_mass = (element_type == 8) | (element_type == 9)
        
        if is_beamsplitter_no_mass:
            # Activate first 4 parameters (loss, reflectivity, tuning, alpha), but not mass
            mask = mask.at[param_idx:param_idx + 4].set(True)
        elif is_beamsplitter_with_mass:
            # Activate all 5 parameters (loss, reflectivity, tuning, alpha, mass)
            mask = mask.at[param_idx:param_idx + 5].set(True)
        # Directional beamsplitters (10-17): no parameters are active
        
        # Move to next beamsplitter's parameter slots (5 params per beamsplitter)
        param_idx += 5
    
    # ===== POSITIONAL ENCODING PARAMETERS =====
    # So far all are active (TODO: will change in the future)
    
    # 5. BOUNDARY SOURCE DISTANCES (4n params)
    # Clockwise from top left: top, right, bottom, left
    mask = mask.at[param_idx:param_idx + 4 * n].set(True)
    param_idx += 4 * n
    
    # 6. ROW SPACING DISTANCES ((n-1) params)
    # Distances between consecutive rows
    mask = mask.at[param_idx:param_idx + (n - 1)].set(True)
    param_idx += (n - 1)
    
    # 7. COLUMN SPACING DISTANCES ((n-1) params)
    # Distances between consecutive columns
    mask = mask.at[param_idx:param_idx + (n - 1)].set(True)
    param_idx += (n - 1)
    
    # 8. BOUNDARY MIRROR RELATIVE DISTANCES (4n params)
    # Clockwise from top left
    mask = mask.at[param_idx:param_idx + 4 * n].set(True)
    param_idx += 4 * n
    
    # 9. CELL MIRROR RELATIVE DISTANCES (4n² params)
    # Order: top, right, bottom, left for each cell, then by row, then by col
    mask = mask.at[param_idx:param_idx + 4 * n * n].set(True)
    param_idx += 4 * n * n
    
    # Sanity check: ensure we've processed all parameters
    assert param_idx == N_parameters, f"Parameter count mismatch: expected {N_parameters}, got {param_idx}"
    
    return mask


def get_physical_parameter_index_map(element_array: jnp.ndarray, n: int) -> dict[tuple[str, str], int]:
    """
    Generate a mapping from (element_name, property_name) to parameter index.
    
    This function accounts for the actual element types in the element_array, so it correctly
    maps laser properties (power, phase) vs squeezer properties (db, angle), and includes
    only parameters for elements that actually exist.
    
    Parameters
    ----------
    element_array : jnp.ndarray
        The element array. Shape: (N_elements,) where N_elements = n * (8 + 5*n).
    n : int
        Grid size (n x n grid)
        
    Returns
    -------
    dict[tuple[str, str], int]
        Dictionary mapping (element_name, property_name) to parameter array index.
        Only includes physical parameters, not positional encoding parameters.
        For elements with free mass, the element name has "sus" appended for the mass property.
        
    Examples
    --------
    >>> element_array = jnp.array([2, 0, ...])  # Laser at boundary01
    >>> param_map = get_physical_parameter_index_map(element_array, n=2)
    >>> param_map[("boundary01", "power")]
    0
    >>> param_map[("boundary01", "phase")]
    1
    """
    param_map = {}
    
    # Helper function to generate boundary coordinate ordering (clockwise from top-left)
    def get_boundary_coords(n: int) -> list[tuple[int, int]]:
        """Returns list of (x, y) coordinates for boundary elements in clockwise order."""
        coords = []
        # Top boundary: x=0, y=1..n
        for y in range(1, n + 1):
            coords.append((0, y))
        # Right boundary: x=1..n, y=n+1
        for x in range(1, n + 1):
            coords.append((x, n + 1))
        # Bottom boundary: x=n+1, y=n..1 (reversed)
        for y in range(n, 0, -1):
            coords.append((n + 1, y))
        # Left boundary: x=n..1 (reversed), y=0
        for x in range(n, 0, -1):
            coords.append((x, 0))
        return coords
    
    # Extract element subsets from the flat element_array
    n_boundary_sources = 4 * n
    n_boundary_mirrors = 4 * n
    n_cell_mirrors = 4 * n * n
    n_cell_beamsplitters = n * n
    
    boundary_sources = element_array[:n_boundary_sources]
    boundary_mirrors = element_array[n_boundary_sources:n_boundary_sources + n_boundary_mirrors]
    cell_mirrors = element_array[n_boundary_sources + n_boundary_mirrors:n_boundary_sources + n_boundary_mirrors + n_cell_mirrors]
    cell_beamsplitters = element_array[n_boundary_sources + n_boundary_mirrors + n_cell_mirrors:]
    
    boundary_coords = get_boundary_coords(n)
    param_idx = 0
    
    # 1. BOUNDARY SOURCE PARAMETERS (2 params per source × 4n sources)
    for i, (x, y) in enumerate(boundary_coords):
        element_type = int(boundary_sources[i])
        element_name = f"boundary{x}{y}"
        
        # Determine parameter names based on element type
        if element_type == 1:  # Detector - no parameters
            pass
        elif element_type == 2:  # Laser
            param_map[(element_name, "power")] = param_idx
            param_map[(element_name, "phase")] = param_idx + 1
        elif element_type == 3:  # Squeezer
            param_map[(element_name, "db")] = param_idx
            param_map[(element_name, "angle")] = param_idx + 1
        
        # Always advance by 2 slots per source
        param_idx += 2
    
    # 2. BOUNDARY MIRROR PARAMETERS (4 params per mirror × 4n mirrors)
    mirror_params = ['loss', 'reflectivity', 'tuning', 'mass']
    
    for i, (x, y) in enumerate(boundary_coords):
        element_type = int(boundary_mirrors[i])
        element_name = f"m{x}{y}"
        
        if element_type == 4:  # Mirror (no free mass)
            for param in mirror_params[:3]:  # loss, reflectivity, tuning
                param_map[(element_name, param)] = param_idx
                param_idx += 1
            param_idx += 1  # Skip mass slot
        elif element_type == 5:  # Mirror with free mass
            for param in mirror_params[:3]:
                param_map[(element_name, param)] = param_idx
                param_idx += 1
            # Mass parameter with "sus" suffix
            param_map[(element_name + "sus", "mass")] = param_idx
            param_idx += 1
        else:  # No element
            param_idx += 4
    
    # 3. CELL MIRROR PARAMETERS (4 params per mirror × 4n² mirrors)
    mirror_directions = ['l', 't', 'r', 'b']
    cell_mirror_idx = 0
    
    for row in range(1, n + 1):
        for col in range(1, n + 1):
            for direction in mirror_directions:
                element_type = int(cell_mirrors[cell_mirror_idx])
                element_name = f"m{direction}{row}{col}"
                
                if element_type == 4:  # Mirror (no free mass)
                    for param in mirror_params[:3]:
                        param_map[(element_name, param)] = param_idx
                        param_idx += 1
                    param_idx += 1  # Skip mass slot
                elif element_type == 5:  # Mirror with free mass
                    for param in mirror_params[:3]:
                        param_map[(element_name, param)] = param_idx
                        param_idx += 1
                    param_map[(element_name + "sus", "mass")] = param_idx
                    param_idx += 1
                else:  # No element
                    param_idx += 4
                
                cell_mirror_idx += 1
    
    # 4. CELL BEAMSPLITTER PARAMETERS (5 params per beamsplitter × n² beamsplitters)
    beamsplitter_params = ['loss', 'reflectivity', 'tuning', 'alpha', 'mass']
    bs_idx = 0
    
    for row in range(1, n + 1):
        for col in range(1, n + 1):
            element_type = int(cell_beamsplitters[bs_idx])
            element_name = f"center{row}{col}"
            
            # Regular beamsplitters: 6-9
            # Directional beamsplitters: 10-17 (no parameters)
            if element_type in [6, 7]:  # Beamsplitter without mass
                for param in beamsplitter_params[:4]:  # loss, reflectivity, tuning, alpha
                    param_map[(element_name, param)] = param_idx
                    param_idx += 1
                param_idx += 1  # Skip mass slot
            elif element_type in [8, 9]:  # Beamsplitter with mass
                for param in beamsplitter_params[:4]:
                    param_map[(element_name, param)] = param_idx
                    param_idx += 1
                param_map[(element_name + "sus", "mass")] = param_idx
                param_idx += 1
            else:  # No element or directional beamsplitter
                param_idx += 5
            
            bs_idx += 1
    
    return param_map



def get_length_indices(connection_name: str, n: int) -> dict:
    """
    Generate index structure for calculating the length of a connection between two elements.
    
    This function returns all indices needed to compute the connection length from a distance array
    in a vectorized manner. This is more efficient than get_length_function() which returns a 
    callable that must be invoked separately for each connection.
    
    Assumes naming convention from (sparse) uifo setup, i.e. '{name}{x}{y}', 
    where mirror names are 'm' for boundary mirrors and 'mt', 'mr', 'mb', 'ml' for cell mirrors.
    
    Parameters
    ----------
    connection_name : str
        Name of the connection, e.g. 'm11_m12' for a connection between mirror 11 and mirror 12. 
        As a convention, the source element is the one before the underscore and the target 
        element is the one after the underscore.
    n : int
        Grid size (n x n grid)

    Returns
    -------
    dict
        Dictionary with the following structure:
        {
            'grid_indices': jnp.ndarray,  # Indices of grid spacing distances to sum
            'slice_start': int,           # Start index for slicing grid_indices
            'slice_end': int,             # End index for slicing grid_indices
            'src_is_mirror': bool,        # Whether source has mirror adjustment
            'src_rel_idx': int,           # Source mirror relative distance index (if is_mirror)
            'src_base_idx': int,          # Source mirror base distance index (if is_mirror)
            'src_sign': float,            # Sign for source mirror adjustment (if is_mirror)
            'tgt_is_mirror': bool,        # Whether target has mirror adjustment
            'tgt_rel_idx': int,           # Target mirror relative distance index (if is_mirror)
            'tgt_base_idx': int,          # Target mirror base distance index (if is_mirror)
            'tgt_sign': float,            # Sign for target mirror adjustment (if is_mirror)
        }
        
    Examples
    --------
    >>> indices = get_length_indices('m11_m12', n=3)
    >>> # Then compute length as:
    >>> grid_sum = distance_array[indices['grid_indices'][indices['slice_start']:indices['slice_end']]].sum()
    >>> src_adj = (indices['src_sign'] * distance_array[indices['src_rel_idx']] * 
    ...            distance_array[indices['src_base_idx']] if indices['src_is_mirror'] else 0.0)
    >>> tgt_adj = (indices['tgt_sign'] * distance_array[indices['tgt_rel_idx']] * 
    ...            distance_array[indices['tgt_base_idx']] if indices['tgt_is_mirror'] else 0.0)
    >>> length = jnp.abs(grid_sum + src_adj + tgt_adj)
    """
    src_name, tgt_name = connection_name.split("_")
    x1, y1, mirror_type_src = _element_name_to_numbers(src_name)
    x2, y2, mirror_type_tgt = _element_name_to_numbers(tgt_name)
    
    return _get_length_indices_from_positions(x1, x2, y1, y2, mirror_type_src, mirror_type_tgt, n)


def compute_length_from_indices(distance_array: jnp.ndarray, indices: dict) -> float:
    """
    Compute a single connection length from a distance array using pre-computed indices.
    
    This is a helper function that shows how to use the indices from get_length_indices()
    to compute the actual length value. For batch computation of multiple connections,
    use compute_lengths_batch() instead.
    
    Parameters
    ----------
    distance_array : jnp.ndarray
        The distance array containing all positional encoding parameters
    indices : dict
        Index structure returned by get_length_indices()
        
    Returns
    -------
    float
        The computed length for this connection
        
    Examples
    --------
    >>> indices = get_length_indices('m11_m12', n=3)
    >>> length = compute_length_from_indices(distance_array, indices)
    """
    # Sum grid spacing distances
    grid_sum = distance_array[indices['grid_indices'][indices['slice_start']:indices['slice_end']]].sum()
    
    # Add source mirror adjustment
    src_adj = 0.0
    if indices['src_is_mirror']:
        src_adj = (indices['src_sign'] * 
                  distance_array[indices['src_rel_idx']] * 
                  distance_array[indices['src_base_idx']])
    
    # Add target mirror adjustment
    tgt_adj = 0.0
    if indices['tgt_is_mirror']:
        tgt_adj = (indices['tgt_sign'] * 
                  distance_array[indices['tgt_rel_idx']] * 
                  distance_array[indices['tgt_base_idx']])
    
    return jnp.abs(grid_sum + src_adj + tgt_adj)


def prepare_batch_indices(all_indices: list[dict]) -> dict:
    """
    Convert a list of index dictionaries into stacked arrays for vectorized computation.
    
    This function takes the output of multiple get_length_indices() calls and prepares
    them for efficient batch computation using compute_lengths_batch().
    
    Parameters
    ----------
    all_indices : list[dict]
        List of index dictionaries from get_length_indices()
        
    Returns
    -------
    dict
        Dictionary with stacked arrays:
        {
            'grid_indices': jnp.ndarray,      # Shape: (n_connections, max_grid_size)
            'grid_mask': jnp.ndarray,          # Shape: (n_connections, max_grid_size), bool
            'src_is_mirror': jnp.ndarray,      # Shape: (n_connections,), bool
            'src_rel_idx': jnp.ndarray,        # Shape: (n_connections,), int
            'src_base_idx': jnp.ndarray,       # Shape: (n_connections,), int
            'src_sign': jnp.ndarray,           # Shape: (n_connections,), float
            'tgt_is_mirror': jnp.ndarray,      # Shape: (n_connections,), bool
            'tgt_rel_idx': jnp.ndarray,        # Shape: (n_connections,), int
            'tgt_base_idx': jnp.ndarray,       # Shape: (n_connections,), int
            'tgt_sign': jnp.ndarray,           # Shape: (n_connections,), float
        }
        
    Examples
    --------
    >>> indices_list = [get_length_indices(conn, n) for conn in connections]
    >>> batch_indices = prepare_batch_indices(indices_list)
    >>> lengths = compute_lengths_batch(distance_array, batch_indices)
    """
    n_connections = len(all_indices)
    
    # Determine max grid size (typically n+2 for row/col indices)
    max_grid_size = max(len(idx['grid_indices']) for idx in all_indices)
    
    # Initialize arrays
    grid_indices = jnp.zeros((n_connections, max_grid_size), dtype=jnp.int32)
    grid_mask = jnp.zeros((n_connections, max_grid_size), dtype=bool)
    
    src_is_mirror = jnp.zeros(n_connections, dtype=bool)
    src_rel_idx = jnp.zeros(n_connections, dtype=jnp.int32)
    src_base_idx = jnp.zeros(n_connections, dtype=jnp.int32)
    src_sign = jnp.zeros(n_connections, dtype=jnp.float32)
    
    tgt_is_mirror = jnp.zeros(n_connections, dtype=bool)
    tgt_rel_idx = jnp.zeros(n_connections, dtype=jnp.int32)
    tgt_base_idx = jnp.zeros(n_connections, dtype=jnp.int32)
    tgt_sign = jnp.zeros(n_connections, dtype=jnp.float32)
    
    # Fill arrays
    for i, idx in enumerate(all_indices):
        # Grid indices and mask
        grid_len = len(idx['grid_indices'])
        grid_indices = grid_indices.at[i, :grid_len].set(idx['grid_indices'])
        
        # Create mask for the slice region
        slice_start = idx['slice_start']
        slice_end = idx['slice_end']
        mask = jnp.arange(max_grid_size)
        mask = (mask >= slice_start) & (mask < slice_end)
        grid_mask = grid_mask.at[i].set(mask)
        
        # Source mirror parameters
        src_is_mirror = src_is_mirror.at[i].set(idx['src_is_mirror'])
        src_rel_idx = src_rel_idx.at[i].set(idx['src_rel_idx'])
        src_base_idx = src_base_idx.at[i].set(idx['src_base_idx'])
        src_sign = src_sign.at[i].set(idx['src_sign'])
        
        # Target mirror parameters
        tgt_is_mirror = tgt_is_mirror.at[i].set(idx['tgt_is_mirror'])
        tgt_rel_idx = tgt_rel_idx.at[i].set(idx['tgt_rel_idx'])
        tgt_base_idx = tgt_base_idx.at[i].set(idx['tgt_base_idx'])
        tgt_sign = tgt_sign.at[i].set(idx['tgt_sign'])
    
    return {
        'grid_indices': grid_indices,
        'grid_mask': grid_mask,
        'src_is_mirror': src_is_mirror,
        'src_rel_idx': src_rel_idx,
        'src_base_idx': src_base_idx,
        'src_sign': src_sign,
        'tgt_is_mirror': tgt_is_mirror,
        'tgt_rel_idx': tgt_rel_idx,
        'tgt_base_idx': tgt_base_idx,
        'tgt_sign': tgt_sign,
    }


def compute_lengths_batch(distance_array: jnp.ndarray, batch_indices: dict) -> jnp.ndarray:
    """
    Compute multiple connection lengths in a single vectorized operation.
    
    This function takes pre-processed batch indices (from prepare_batch_indices) and
    computes all connection lengths at once without Python loops or if statements.
    
    Parameters
    ----------
    distance_array : jnp.ndarray
        The distance array containing all positional encoding parameters
    batch_indices : dict
        Batch index structure from prepare_batch_indices()
        
    Returns
    -------
    jnp.ndarray
        Array of computed lengths, shape: (n_connections,)
        
    Examples
    --------
    >>> indices_list = [get_length_indices(conn, n) for conn in connections]
    >>> batch_indices = prepare_batch_indices(indices_list)
    >>> lengths = compute_lengths_batch(distance_array, batch_indices)
    """
    # Extract batch indices
    grid_indices = batch_indices['grid_indices']
    grid_mask = batch_indices['grid_mask']
    
    src_is_mirror = batch_indices['src_is_mirror']
    src_rel_idx = batch_indices['src_rel_idx']
    src_base_idx = batch_indices['src_base_idx']
    src_sign = batch_indices['src_sign']
    
    tgt_is_mirror = batch_indices['tgt_is_mirror']
    tgt_rel_idx = batch_indices['tgt_rel_idx']
    tgt_base_idx = batch_indices['tgt_base_idx']
    tgt_sign = batch_indices['tgt_sign']
    
    # Gather all grid distance values: (n_connections, max_grid_size)
    grid_distances = distance_array[grid_indices]
    
    # Apply mask and sum: (n_connections,)
    # Only sum distances where mask is True
    grid_sums = jnp.sum(grid_distances * grid_mask, axis=1)
    
    # Compute source mirror adjustments: (n_connections,)
    src_rel_distances = distance_array[src_rel_idx]
    src_base_distances = distance_array[src_base_idx]
    src_adjustments = src_sign * src_rel_distances * src_base_distances
    # Zero out adjustments where there's no mirror
    src_adjustments = jnp.where(src_is_mirror, src_adjustments, 0.0)
    
    # Compute target mirror adjustments: (n_connections,)
    tgt_rel_distances = distance_array[tgt_rel_idx]
    tgt_base_distances = distance_array[tgt_base_idx]
    tgt_adjustments = tgt_sign * tgt_rel_distances * tgt_base_distances
    # Zero out adjustments where there's no mirror
    tgt_adjustments = jnp.where(tgt_is_mirror, tgt_adjustments, 0.0)
    
    # Combine and take absolute value: (n_connections,)
    lengths = jnp.abs(grid_sums + src_adjustments + tgt_adjustments)
    
    return lengths


def get_length_function(connection_name: str, n: int):
    """
    Generate a function that calculates the length of a connection between two elements.
    Assumes naming convention from (sparse) uifo setup, i.e. '{name}{x}{y}', 
    where mirror names are 'm' for boundary mirrors and 'mt', 'mr', 'mb', 'ml' for cell mirrors (top, right, bottom, left).
    
    Parameters
    ----------
    connection_name : str
        Name of the connection, e.g. 'm11_m12' for a connection between mirror 11 and mirror 12. 
        As a convention, the source element is the one before the underscore and the target element is the one after the underscore.
    n : int
        Grid size (n x n grid)

    Returns
    -------
    function
        Function that calculates the length of a connection between two elements given a distance array.
    """
    src_name, tgt_name = connection_name.split("_")
    x1, y1, mirror_type_src = _element_name_to_numbers(src_name)
    x2, y2, mirror_type_tgt = _element_name_to_numbers(tgt_name)
    return _get_length_from_distance_array_fn(x1, x2, y1, y2, mirror_type_src, mirror_type_tgt, n)



def get_positional_encoding_names(n: int) -> tuple[str, ...]:
    """
    Generate parameter names for the positional encoding distance array.
    
    The distance array has the following structure:
    - n*4 boundary source distances (clockwise starting from top left)
    - (n-1) row spacing distances (distances between consecutive rows)
    - (n-1) col spacing distances (distances between consecutive columns)
    - n*4 boundary mirror relative distances (clockwise starting from top left)
    - n*n*4 cell mirror relative distances (top-right-bottom-left, then by row, then by col)
    
    Total length: 8*n + 2*(n-1) + 4*n*n
    
    Parameters
    ----------
    n : int
        Grid size (n x n grid)
        
    Returns
    -------
    tuple[str, ...]
        Tuple of parameter names in the order they appear in the distance array
        
    Examples
    --------
    >>> names = get_positional_encoding_names(3)
    >>> len(names)
    64
    """
    names: list[str] = []
    
    # 1. Boundary source distances (n*4) - clockwise from top left
    # Top boundary: (x=1..n, y=0)
    for x in range(1, n + 1):
        names.append(f"boundary_dist_{x}0")
    
    # Right boundary: (x=n+1, y=1..n)
    for y in range(1, n + 1):
        names.append(f"boundary_dist_{n+1}{y}")
    
    # Bottom boundary: (x=n..1, y=n+1)
    for x in range(n, 0, -1):
        names.append(f"boundary_dist_{x}{n+1}")
    
    # Left boundary: (x=0, y=n..1)
    for y in range(n, 0, -1):
        names.append(f"boundary_dist_0{y}")
    
    # 2. Row spacing distances (n-1)
    for i in range(n - 1):
        names.append(f"row_spacing_{i}{i+1}")
    
    # 3. Column spacing distances (n-1)
    for i in range(n - 1):
        names.append(f"col_spacing_{i}{i+1}")
    
    # 4. Boundary mirror relative distances (n*4) - same ordering as boundary sources
    # Top boundary: (x=1..n, y=0)
    for x in range(1, n + 1):
        names.append(f"boundary_mirror_rel_dist_{x}0")
    
    # Right boundary: (x=n+1, y=1..n)
    for y in range(1, n + 1):
        names.append(f"boundary_mirror_rel_dist_{n+1}{y}")
    
    # Bottom boundary: (x=n..1, y=n+1)
    for x in range(n, 0, -1):
        names.append(f"boundary_mirror_rel_dist_{x}{n+1}")
    
    # Left boundary: (x=0, y=n..1)
    for y in range(n, 0, -1):
        names.append(f"boundary_mirror_rel_dist_0{y}")

    
    # 5. Cell mirror relative distances (n*n*4)
    # Order: top, right, bottom, left for each cell
    # Then iterate: rows left to right, then columns top to bottom
    mirror_directions = ['t', 'r', 'b', 'l']
    for y in range(1, n + 1):  # column index (top to bottom)
        for x in range(1, n + 1):  # row index (left to right)
            for direction in mirror_directions:
                names.append(f"cell_mirror_rel_dist_m{direction}{x}{y}")
    
    return tuple(names)

#-------------------------------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


def _get_length_indices_from_positions(x1: int, x2: int, y1: int, y2: int, 
                                       mirror_type_src: int, mirror_type_tgt: int, n: int) -> dict:
    """
    Extract all indices needed to compute a connection length from a distance array.
    
    This is the core function that computes the index structure for vectorized length calculation.
    """
    # Determine direction (horizontal=0, vertical=1)
    direction = _get_direction(x1, x2, mirror_type_src, mirror_type_tgt, n)
    direction_int = int(direction)
    
    # Get the full row/column distance indices array
    if direction_int == 0:  # Horizontal
        grid_indices = _get_row_distance_indices(y1, n)
        slice_coords = jnp.array([x1, x2])
    else:  # Vertical
        grid_indices = _get_col_distance_indices(x1, n)
        slice_coords = jnp.array([y1, y2])
    
    # Compute slice boundaries for grid spacing sum
    sorted_coords = jnp.sort(slice_coords)
    slice_start = int(sorted_coords[0])
    slice_end = int(sorted_coords[1])
    
    # Initialize result dictionary
    result = {
        'grid_indices': grid_indices,
        'slice_start': slice_start,
        'slice_end': slice_end,
    }
    
    # Process source mirror adjustment
    if mirror_type_src == 0:  # No mirror
        result['src_is_mirror'] = False
        result['src_rel_idx'] = 0
        result['src_base_idx'] = 0
        result['src_sign'] = 0.0
    elif mirror_type_src == 1:  # Boundary mirror
        result['src_is_mirror'] = True
        result['src_rel_idx'] = int(_get_boundary_mirror_indices(x1, y1, n))
        result['src_base_idx'] = int(_get_boundary_mirror_base_indices(x1, y1, n))
        result['src_sign'] = -1.0
    else:  # Cell mirror (types 2-5)
        result['src_is_mirror'] = True
        result['src_rel_idx'] = int(_get_cell_mirror_indices(x1, y1, n, mirror_type_src))
        
        # Compute base index from grid_indices array
        coord = (1 - direction_int) * x1 + direction_int * y1
        offset = -1 if (mirror_type_src == 2 or mirror_type_src == 5) else 0
        idx = coord + offset
        result['src_base_idx'] = int(grid_indices[idx])
        
        # Compute sign for cell mirror
        sign = -1.0  # is_source=True
        if mirror_type_src == 2 or mirror_type_src == 5:  # Top or left
            sign = 1.0
        result['src_sign'] = sign
    
    # Process target mirror adjustment
    if mirror_type_tgt == 0:  # No mirror
        result['tgt_is_mirror'] = False
        result['tgt_rel_idx'] = 0
        result['tgt_base_idx'] = 0
        result['tgt_sign'] = 0.0
    elif mirror_type_tgt == 1:  # Boundary mirror
        result['tgt_is_mirror'] = True
        result['tgt_rel_idx'] = int(_get_boundary_mirror_indices(x2, y2, n))
        result['tgt_base_idx'] = int(_get_boundary_mirror_base_indices(x2, y2, n))
        result['tgt_sign'] = -1.0
    else:  # Cell mirror (types 2-5)
        result['tgt_is_mirror'] = True
        result['tgt_rel_idx'] = int(_get_cell_mirror_indices(x2, y2, n, mirror_type_tgt))
        
        # Compute base index from grid_indices array
        coord = (1 - direction_int) * x2 + direction_int * y2
        offset = -1 if (mirror_type_tgt == 2 or mirror_type_tgt == 5) else 0
        idx = coord + offset
        result['tgt_base_idx'] = int(grid_indices[idx])
        
        # Compute sign for cell mirror
        sign = 1.0  # is_source=False
        if mirror_type_tgt == 2 or mirror_type_tgt == 5:  # Top or left
            sign = -1.0
        result['tgt_sign'] = sign
    
    return result


def _element_name_to_numbers(name: str) -> tuple[int, int, int]:
    y, x = map(int, name[-2:])  # we actually use {vertical}{horizontal} convention in the setup ("vertical" corresponds to top/bottom mirrors)
    mirror_type = _get_mirror_type_from_name(name[:-2])
    return x, y, mirror_type


def _get_mirror_type_from_name(name: str) -> int:
    # mirror: 0 (no), 1 (boundary), 2 (cell top), 3 (cell right), 4 (cell bottom), 5 (cell left)
    mirror_type = 0
    mirror_type = jnp.where(name == 'm', 1, mirror_type)
    mirror_type = jnp.where(name == 'mt', 2, mirror_type)
    mirror_type = jnp.where(name == 'mr', 3, mirror_type)
    mirror_type = jnp.where(name == 'mb', 4, mirror_type)
    mirror_type = jnp.where(name == 'ml', 5, mirror_type)

    return mirror_type


def _get_row_distance_indices(y: int, n: int):
    # assume y in [1, n]
    return jnp.array([4 * n - y] + [n * 4 + i for i in range(n - 1)] + [n + y - 1])


def _get_col_distance_indices(x: int, n: int):
    # assume x in [1, n]
    return jnp.array([x - 1] + [n * 4 + n - 1 + i for i in range(n - 1)] + [3 * n  - x])


def _get_boundary_mirror_indices(x: int, y: int, n: int):
    start = n * 4 + 2 * (n - 1)
    
    return _get_boundary_offset_indices(x, y, n) + start


def _get_boundary_mirror_base_indices(x: int, y: int, n: int):
    # start = 0
    return _get_boundary_offset_indices(x, y, n)


def _get_boundary_offset_indices(x: int, y: int, n: int):    
    offset = jnp.array(0)

    # unroll clockwise starting from top left corner
    offset = jnp.where(y == 0, x - 1, offset)
    offset = jnp.where(x == n + 1, y - 1 + n, offset)
    offset = jnp.where(y == n + 1, 3 * n  - x, offset)
    offset = jnp.where(x == 0, 4 * n  - y, offset)

    return offset


def _get_boundary_mirror_length_fn(x: int, y: int, mirror_type: int, n: int):
    rel_distance_idx = _get_boundary_mirror_indices(x, y, n)
    base_distance_idx = _get_boundary_mirror_base_indices(x, y, n)

    def add_length_fn(distance_array):
        return - distance_array[rel_distance_idx] * distance_array[base_distance_idx]
    return add_length_fn


def _get_cell_mirror_length_fn(x: int, y: int, mirror_type: int, n: int, row_indices: jnp.ndarray, direction: int, is_source: bool):
    sign = jnp.array(1) * (-1 if is_source else 1)

    # if a mirror is a target, then top and left mirrors should have a negative sign.
    # otherwise, they should have a positive sign (and vice versa for other mirrors)
    sign = jnp.where(mirror_type == 2, -sign, sign)
    sign = jnp.where(mirror_type == 5, -sign, sign)

    rel_distance_idx = _get_cell_mirror_indices(x, y, n, mirror_type)
    x = (1 - direction) * x + direction * y
    base_distance_idx = _get_cell_mirror_base_indices_from_row(row_indices, x, mirror_type)

    def add_length_fn(distance_array):
        return sign * distance_array[base_distance_idx] * distance_array[rel_distance_idx]
    
    return add_length_fn


def _get_zero_add_length_fn(*args):
    def zero_add_length(distance_array):
        return 0
    return zero_add_length


def _get_cell_mirror_indices(x: int, y: int, n: int, mirror_type: int):
    # assume x, y in [1, n]
    
    start = n * 8 + 2 * (n - 1)
    offset = _get_cell_mirror_offset(mirror_type)
    
    return start + (x - 1) * 4 + (y - 1) * n * 4 + offset    


def _get_cell_mirror_base_indices_from_row(row_indices: jnp.ndarray, x: int, mirror_type: int):
    offset = jnp.array(0)

    # minus one for left and top
    offset = jnp.where(mirror_type == 2, -1, offset)
    offset = jnp.where(mirror_type == 5, -1, offset)

    idx = x + offset

    return row_indices[idx]


def _get_cell_mirror_offset(mirror_type: int):
    return mirror_type - 2


def _get_length_from_distance_array_fn(x1: int, x2: int, y1: int, y2: int, mirror_type_src: int, mirror_type_tgt: int, n: int):
    direction_indices = _get_direction(x1, x2, mirror_type_src, mirror_type_tgt, n)

    direction_functions = [
        partial(_get_row_distance_indices, y1, n),
        partial(_get_col_distance_indices, x1, n),
    ]

    slice_indices = jnp.sort(jnp.where(direction_indices, jnp.array([y1, y2]), jnp.array([x1, x2])))

    row = jax.lax.switch(direction_indices, direction_functions)
    
    row_to_sum_indices = row[slice_indices[0]:slice_indices[1]]

    if mirror_type_src == 0:
        add_length_fn_src = _get_zero_add_length_fn()
    elif mirror_type_src == 1:
        add_length_fn_src = _get_boundary_mirror_length_fn(x1, y1, mirror_type_src, n)
    else:
        add_length_fn_src = _get_cell_mirror_length_fn(x1, y1, mirror_type_src, n, row, direction_indices, is_source=True)
    
    if mirror_type_tgt == 0:
        add_length_fn_tgt = _get_zero_add_length_fn()
    elif mirror_type_tgt == 1:
        add_length_fn_tgt = _get_boundary_mirror_length_fn(x2, y2, mirror_type_tgt, n)
    else: 
        add_length_fn_tgt = _get_cell_mirror_length_fn(x2, y2, mirror_type_tgt, n, row, direction_indices, is_source=False)

    def calculate_length_from_distance_array(distance_array):
        return jnp.abs(distance_array[row_to_sum_indices].sum() + add_length_fn_src(distance_array) + add_length_fn_tgt(distance_array))

    return calculate_length_from_distance_array

    
def _get_direction(x1: int, x2: int, mirror_type_src: int, mirror_type_tgt: int, n: int):
    horizontal = (
        (x1 != x2)    |
        (x1 == 0)     | 
        (x1 == n + 1) |
        (x2 == 0)     | 
        (x2 == n + 1) |
        (mirror_type_src == 5) | 
        (mirror_type_src == 3) |
        (mirror_type_tgt == 5) | 
        (mirror_type_tgt == 3)
    )
    
    direction = jnp.where(horizontal, 0, 1)  # 0 for horizontal, 1 for vertical
    
    return direction
