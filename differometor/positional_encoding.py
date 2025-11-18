import jax
import jax.numpy as jnp
from functools import partial


def get_length_function(src_name: str, tgt_name: str, n: int):
    """
    Generate a function that calculates the length of a connection between two elements.
    Assumes naming convention from (sparse) uifo setup, i.e. '{name}{x}{y}', 
    where mirror names are 'm' for boundary mirrors and 'mt', 'mr', 'mb', 'ml' for cell mirrors (top, right, bottom, left).
    
    Parameters
    ----------
    src_name : str
        Name of the source element.
    tgt_name : str
        Name of the target element
    n : int
        Grid size (n x n grid)

    Returns
    -------
    function
        Function that calculates the length of a connection between two elements given a distance array.
    """
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
