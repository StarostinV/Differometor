import jax.numpy as jnp
from differometor.setups import Setup
from differometor.components import DEFAULT_PROPERTIES



def sparse_uifo(
    size: int,
    element_array: jnp.ndarray,
    mode: str = 'space_modulation',
):
    """
    Defines a sparse quasi-universal interferometer (UIFO) based on an element array with default parameters.

    The 1D integer element array defines what elements are present in the UIFO. All elements use default 
    parameters from DEFAULT_PROPERTIES. Here is the structure of the element array:
    
    Integer to element mapping:

    0: no element
    1: detector
    2: laser (default: power=1.0, phase=0.0)
    3: squeezer (default: db=0, angle=90)
    4: mirror (default: loss=5e-6, reflectivity=0.5, tuning=0.0)
    5: mirror with free mass (default: + mass=40.0)
    6: beamsplitter in direction of left port (default: loss=5e-6, reflectivity=0.5, tuning=0.0, alpha=45)
    7: beamsplitter in direction of top port
    8: beamsplitter in direction of left port with free mass (default: + mass=40.0)
    9: beamsplitter in direction of top port with free mass
    10: directional beamsplitter in direction of left port
    11: directional beamsplitter in direction of top port
    12: directional beamsplitter in direction of right port
    13: directional beamsplitter in direction of bottom port
    14: directional beamsplitter in direction of left port with free mass
    15: directional beamsplitter in direction of top port with free mass
    16: directional beamsplitter in direction of right port with free mass
    17: directional beamsplitter in direction of bottom port with free mass

    The max number of elements where n is the size of the UIFO:
    - 1 detector (necessary for the scheme to work; so far we don't support balanced homodyne detection)
    - n * 4 - 1 lasers (-1 because of the detector)
    - n * 4 - 1 squeezers
    - n * 4 + 4 * n * n mirrors (n * 4 for the boundaries and 4 * n * n for the centers)
    - n * n - beamsplitters / directional beamsplitters

    0:n*4 - boundary sources (N_sources=n*4);
        elements: (0: nothing, 1: detector, 2: laser, 3: squeezer)
        order: clockwise starting from top left.
    n*4:8*n - boundary mirrors (N_boundary_mirrors=n*4); 
        elements: (0: nothing, 4: mirror, 5: mirror with free mass)
        order: clockwise starting from top left.
    8*n:8*n + 4*n*n - cell mirrors (N_cell_mirrors=4*n*n);
        elements: (0: nothing, 4: mirror, 5: mirror with free mass)
        order: cell first (clockwise starting from top left), then rows (from left to right), then columns (from top to bottom).
    8*n + 4*n*n:8*n + 5*n*n - cell beamsplitters / directional beamsplitters (N_cell_beamsplitters=n*n);
        elements: (0: nothing, 6-17: beamsplitters)
        order: cell first (clockwise starting from top left), then rows (from left to right), then columns (from top to bottom).

    Total number of element positions: N_elements = N_sources + N_boundary_mirrors + N_cell_mirrors + N_cell_beamsplitters = n*4 + 4*n + 4*n*n + n*n = n * (8 + 5*n).
        N_elements(n=3) = 69; N_elements(n=4) = 112; N_elements(n=5) = 165.

    To resolve invariances in the setup, we allow the detector to be placed only at the one of top left [(n + 1) / 2] boundary positions.

    All spaces use default lengths (1.0) and refractive indices from DEFAULT_PROPERTIES. The grid structure
    is arranged with uniform spacing between elements.

    Parameters
    ----------
    size: int
        The size of the grid. E.g. 3 results in a 3x3 grid.
    element_array: jnp.ndarray
        The element array. Shape: (N_elements,) where N_elements = n * (8 + 5*n).
    mode: str
        Modulation mode: 'space_modulation' (default), 'amplitude_modulation', or 'frequency_modulation'.
    """

    # Calculate the number of elements
    n = size
    n_elements = n * (8 + 5*n)

    # Validate dimensions
    if element_array.shape != (n_elements,):
        raise ValueError(f"Element array must have shape ({n_elements},).")
    
    # Validate mode
    if mode not in ['space_modulation', 'amplitude_modulation', 'frequency_modulation']:
        raise ValueError("Invalid mode. Choose from 'space_modulation', 'amplitude_modulation', or 'frequency_modulation'.")

    # Element type mapping
    element_mapping = {
        0: None,
        1: "detector",
        2: "laser",
        3: "squeezer",
        4: "mirror",
        5: "mirror",  # with free mass
        6: "beamsplitter",  # left port
        7: "beamsplitter",  # top port
        8: "beamsplitter",  # left port with free mass
        9: "beamsplitter",  # top port with free mass
        10: "directional_beamsplitter",  # left port
        11: "directional_beamsplitter",  # top port
        12: "directional_beamsplitter",  # right port
        13: "directional_beamsplitter",  # bottom port
        14: "directional_beamsplitter",  # left port with free mass
        15: "directional_beamsplitter",  # top port with free mass
        16: "directional_beamsplitter",  # right port with free mass
        17: "directional_beamsplitter",  # bottom port with free mass
    }

    # Beamsplitter orientation mapping
    bs_orientation_mapping = {
        6: "left", 7: "top", 8: "left", 9: "top",
        10: "left", 11: "top", 12: "right", 13: "bottom",
        14: "left", 15: "top", 16: "right", 17: "bottom"
    }

    # Extract elements using helper function
    elements = _extract_elements(element_array, n)
    
    # Use default spacing for grid
    default_spacing = 1.0
    
    # Calculate absolute grid center positions with uniform spacing
    row_positions = jnp.arange(n, dtype=float) * default_spacing
    col_positions = jnp.arange(n, dtype=float) * default_spacing
    
    # Initialize rows and cols to store elements with their positions
    # Each element in row/col is a tuple: (name, port_in, port_out, position)
    rows = [[] for _ in range(n)]
    cols = [[] for _ in range(n)]

    # Create the Setup object
    S = Setup()
    
    # Add frequency component (required for signal generation)
    S.add("frequency", "f")
    
    # Default values for spaces and mirrors
    default_boundary_distance = 1.0
    default_mirror_relative_distance = 0.5  # Middle point between grid center and boundary
    
    # Add boundary sources and mirrors
    # Boundary ordering: clockwise from top-left
    # For n=3: top row (0,1), (0,2), (0,3), right col (1,3), (2,3), (3,3), 
    #          bottom row (3,2), (3,1), (3,0), left col (2,0), (1,0), (0,0)
    # But we use simpler indexing: side index 0-3 (top, right, bottom, left), position 0 to n-1
    
    boundary_info = _prepare_boundary_info(n) # [(side, pos, x, y, is_row, idx_in_row_col)]
        
    # Dictionary to track which boundary positions have sources
    boundary_sources = {}
    
    # Process boundary sources
    for boundary_idx, (side, pos_idx, x, y, is_row, idx_in_row_col) in enumerate(boundary_info):
        source_element = elements['source_elements'][boundary_idx]
        
        if source_element == 0:  # No element
            continue
        
        # Add source element using helper function with default parameters
        source_name = f"boundary{x}{y}"
        source_type = _add_source_element_default(
            S, source_element, source_name, element_mapping
        )
        
        if source_type:
            boundary_sources[boundary_idx] = (source_type, source_name)
            
            # Add amplitude or frequency modulation signals for lasers
            if source_type == "laser":
                if mode == "amplitude_modulation":
                    S.add("signal", f"s{source_name}", target=f"{source_name}_amplitude", 
                          amplitude=(f"{source_name}_power", jnp.sqrt))
                elif mode == "frequency_modulation":
                    S.add("signal", f"s{source_name}", target=f"{source_name}_frequency")
    
    # Process boundary mirrors
    for boundary_idx, (side, pos_idx, x, y, is_row, idx_in_row_col) in enumerate(boundary_info):
        mirror_element = elements['boundary_mirrors'][boundary_idx]
        
        if mirror_element == 0:  # No mirror
            continue
        
        # Add boundary mirror using helper function with default parameters
        mirror_name = f"m{x}{y}"
        _add_mirror_element_default(S, mirror_element, mirror_name)
        
        # Calculate mirror position using default distances
        boundary_dist = default_boundary_distance
        mirror_rel_dist = default_mirror_relative_distance
        
        # Check if there's a source at this boundary position
        source_info = boundary_sources.get(boundary_idx, None)
        
        # Connect source to mirror if present
        if source_info:
            source_type, source_name = source_info
            _connect_source_to_mirror(S, source_type, source_name, mirror_name)
        
        # Position calculation and adding to rows/cols depends on which side
        if side == "top":
            # Column idx_in_row_col, positioned above the grid
            grid_center_row = row_positions[0]
            mirror_position = grid_center_row - boundary_dist - mirror_rel_dist
            # Add mirror to column (mirrors face the grid center with their right port)
            cols[idx_in_row_col].append((mirror_name, "right", "left", mirror_position))
            
        elif side == "bottom":
            # Column idx_in_row_col, positioned below the grid
            grid_center_row = row_positions[n - 1]
            mirror_position = grid_center_row + boundary_dist + mirror_rel_dist
            # Add mirror to column
            cols[idx_in_row_col].append((mirror_name, "right", "left", mirror_position))
            
        elif side == "left":
            # Row idx_in_row_col, positioned left of the grid
            grid_center_col = col_positions[0]
            mirror_position = grid_center_col - boundary_dist - mirror_rel_dist
            # Add mirror to row (mirrors face the grid center with their right port)
            rows[idx_in_row_col].append((mirror_name, "right", "left", mirror_position))
            
        elif side == "right":
            # Row idx_in_row_col, positioned right of the grid
            grid_center_col = col_positions[n - 1]
            mirror_position = grid_center_col + boundary_dist + mirror_rel_dist
            # Add mirror to row
            rows[idx_in_row_col].append((mirror_name, "right", "left", mirror_position))
    
    # Process cell beamsplitters and mirrors
    for row_idx in range(n):
        for col_idx in range(n):
            # Get beamsplitter element
            bs_element = int(elements['beamsplitters'][row_idx, col_idx])
             # Get grid center position
            grid_row_pos = row_positions[row_idx]
            grid_col_pos = col_positions[col_idx]
            
            if bs_element != 0:  # No beamsplitter
            
                # Add beamsplitter using helper function with default parameters
                bs_name = f"center{row_idx + 1}{col_idx + 1}"
                bs_type, orientation = _add_beamsplitter_element_default(
                    S, bs_element, bs_name,
                    element_mapping, bs_orientation_mapping
                )
       
                # Add beamsplitter to both row and col
                # The ports depend on orientation
                if orientation == "left":
                    rows[row_idx].append((bs_name, "left", "right", grid_col_pos))
                    cols[col_idx].append((bs_name, "top", "bottom", grid_row_pos))
                elif orientation == "top":
                    rows[row_idx].append((bs_name, "top", "bottom", grid_col_pos))
                    cols[col_idx].append((bs_name, "right", "left", grid_row_pos))
                elif orientation == "right":
                    rows[row_idx].append((bs_name, "right", "left", grid_col_pos))
                    cols[col_idx].append((bs_name, "left", "right", grid_row_pos))
                elif orientation == "bottom":
                    rows[row_idx].append((bs_name, "bottom", "top", grid_col_pos))
                    cols[col_idx].append((bs_name, "left", "right", grid_row_pos))
            
            # Process cell mirrors (4 mirrors per cell: left, top, right, bottom)
            mirror_names = [f"ml{row_idx + 1}{col_idx + 1}", f"mt{row_idx + 1}{col_idx + 1}", 
                          f"mr{row_idx + 1}{col_idx + 1}", f"mb{row_idx + 1}{col_idx + 1}"]
            
            for mirror_idx, mirror_local_name in enumerate(mirror_names):
                cell_mirror_element = int(elements['cell_mirrors'][row_idx, col_idx, mirror_idx])
                
                if cell_mirror_element == 0:  # No mirror
                    continue
                
                # Add cell mirror using helper function with default parameters
                _add_mirror_element_default(S, cell_mirror_element, mirror_local_name)
                
                # Calculate mirror position using default relative distance
                mirror_rel_dist = default_mirror_relative_distance
                
                # Mirror 0 (left), Mirror 2 (right) go in rows
                # Mirror 1 (top), Mirror 3 (bottom) go in columns
                if mirror_idx == 0:  # Left mirror
                    mirror_position = grid_col_pos - mirror_rel_dist
                    rows[row_idx].append((mirror_local_name, "right", "left", mirror_position))
                elif mirror_idx == 1:  # Top mirror
                    mirror_position = grid_row_pos - mirror_rel_dist
                    cols[col_idx].append((mirror_local_name, "right", "left", mirror_position))
                elif mirror_idx == 2:  # Right mirror
                    mirror_position = grid_col_pos + mirror_rel_dist
                    rows[row_idx].append((mirror_local_name, "right", "left", mirror_position))
                elif mirror_idx == 3:  # Bottom mirror
                    mirror_position = grid_row_pos + mirror_rel_dist
                    cols[col_idx].append((mirror_local_name, "right", "left", mirror_position))
    
    # Now connect elements within each row and column
    # Sort elements by position and connect them with spaces
    
    for row_idx, row in enumerate(rows):
        if len(row) < 2:
            continue
        
        # Sort by position
        row_sorted = sorted(row, key=lambda x: x[3])
        
        # Connect adjacent elements
        for i in range(len(row_sorted) - 1):
            src_name, src_port_in, src_port_out, src_pos = row_sorted[i]
            tgt_name, tgt_port_in, tgt_port_out, tgt_pos = row_sorted[i + 1]
            
            # Calculate space length
            length = abs(tgt_pos - src_pos)
            
            # Add space
            S.space(src_name, tgt_name, length=length, 
                   source_port=src_port_out, target_port=tgt_port_in)
            
            # Add signal for horizontal space (phase 0) - only for space_modulation mode
            if mode == "space_modulation":
                S.add("signal", f"s{src_name}{tgt_name}", target=f"{src_name}_{tgt_name}", phase=0)
    
    for col_idx, col in enumerate(cols):
        if len(col) < 2:
            continue
        
        # Sort by position
        col_sorted = sorted(col, key=lambda x: x[3])
        
        # Connect adjacent elements
        for i in range(len(col_sorted) - 1):
            src_name, src_port_in, src_port_out, src_pos = col_sorted[i]
            tgt_name, tgt_port_in, tgt_port_out, tgt_pos = col_sorted[i + 1]
            
            # Calculate space length
            length = abs(tgt_pos - src_pos)
            
            # Add space
            S.space(src_name, tgt_name, length=length,
                   source_port=src_port_out, target_port=tgt_port_in)
            
            # Add signal for vertical space (phase 180) - only for space_modulation mode
            if mode == "space_modulation":
                S.add("signal", f"s{src_name}{tgt_name}", target=f"{src_name}_{tgt_name}", phase=180)
    
    # Return the setup and parameters
    return S, S.parameters


def prune_disconnected_components(
    setup: Setup,
    size: int = None,
    element_array: jnp.ndarray = None
) -> dict:
    """
    Remove all connected components except the one containing the detector.
    
    This function analyzes the graph structure of the setup, identifies all connected
    components, and removes all components that don't contain a detector node. It also
    removes any signals (including space modulation signals) associated with removed
    spaces.
    
    Note: This function currently only supports simple detector components. Balanced
    homodyne detection (qhd) is not yet supported.
    
    Parameters
    ----------
    setup : Setup
        The setup object to prune
    size : int, optional
        The grid size (required if element_array is provided)
    element_array : jnp.ndarray, optional
        The original element array. If provided, an updated array with zeros for
        removed elements will be returned.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'modified': bool - True if any components were removed
        - 'updated_element_array': jnp.ndarray - Only present if element_array was provided.
          The updated array with zeros for removed elements.
          
    Examples
    --------
    >>> element_array = jnp.array([...])
    >>> setup, _ = sparse_uifo(size=3, element_array=element_array)
    >>> result = prune_disconnected_components(setup, size=3, element_array=element_array)
    >>> if result['modified']:
    ...     print(f"Removed disconnected components")
    ...     updated_array = result['updated_element_array']
    """
    from collections import deque
    
    # Build adjacency list from edges and target connections
    adjacency = {}
    for node in setup._nodes:
        adjacency[node] = set()
    
    # Add edges (bidirectional since the graph is undirected for connectivity purposes)
    for (src, tgt) in setup._edges:
        if src in adjacency and tgt in adjacency:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
    
    # Add target connections (some components like detectors, free_mass use target parameter)
    for node_name, node_info in setup._nodes.items():
        target = node_info.get('target')
        if target and target in adjacency:
            adjacency[node_name].add(target)
            adjacency[target].add(node_name)
    
    # Find all detector nodes
    detector_nodes = set()
    for node_name, node_info in setup._nodes.items():
        if node_info['component'] == 'detector':
            detector_nodes.add(node_name)
    
    if not detector_nodes:
        # No detector found, don't modify anything
        result = {'modified': False}
        if element_array is not None:
            result['updated_element_array'] = element_array
        return result
    
    # Find connected component containing detector using BFS
    detector_component = set()
    queue = deque(detector_nodes)
    visited = set(detector_nodes)
    
    while queue:
        node = queue.popleft()
        if node not in adjacency:
            continue
        detector_component.add(node)
        
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Identify nodes to remove (all nodes not in detector component)
    all_nodes = set(setup._nodes.keys())
    nodes_to_remove = all_nodes - detector_component
    
    # Check if anything needs to be removed
    modified = len(nodes_to_remove) > 0
    
    if not modified:
        result = {'modified': False}
        if element_array is not None:
            result['updated_element_array'] = element_array
        return result
    
    # Build mapping from element names to positions in element_array if provided
    updated_array = None
    if element_array is not None and size is not None:
        updated_array = element_array.copy()
        # We need to map removed nodes back to element array positions
        removed_elements = _map_nodes_to_element_positions(nodes_to_remove, size)
        # Zero out removed elements
        for pos in removed_elements:
            updated_array = updated_array.at[pos].set(0)
    
    # Remove nodes from setup
    for node in nodes_to_remove:
        if node in setup._nodes:
            del setup._nodes[node]
    
    # Remove edges connected to removed nodes
    edges_to_remove = []
    for (src, tgt) in setup._edges:
        if src in nodes_to_remove or tgt in nodes_to_remove:
            edges_to_remove.append((src, tgt))
    
    for edge in edges_to_remove:
        if edge in setup._edges:
            del setup._edges[edge]
    
    # Remove signals associated with removed spaces or components
    signals_to_remove = []
    for node_name, node_info in list(setup._nodes.items()):
        if node_info['component'] == 'signal':
            target = node_info.get('target')
            if target:
                # Check if target ends with special suffixes (amplitude/frequency modulation)
                if target.endswith('_amplitude') or target.endswith('_frequency'):
                    # Extract the component name (remove suffix)
                    component_name = target.rsplit('_', 1)[0]
                    if component_name in nodes_to_remove:
                        signals_to_remove.append(node_name)
                elif '_' in target:
                    # This might be a space signal with target "node1_node2"
                    parts = target.split('_', 1)
                    if len(parts) == 2:
                        src, tgt = parts
                        # Check if this is a space between two components
                        # (both components should exist in the setup or be removed)
                        if src in all_nodes or tgt in all_nodes:
                            # This is a space signal
                            if src in nodes_to_remove or tgt in nodes_to_remove:
                                signals_to_remove.append(node_name)
                        else:
                            # Might be an amplitude/frequency target, check if base component removed
                            if target.split('_')[0] in nodes_to_remove:
                                signals_to_remove.append(node_name)
                elif target in nodes_to_remove:
                    # Signal targets a removed component directly
                    signals_to_remove.append(node_name)
    
    for signal in signals_to_remove:
        if signal in setup._nodes:
            del setup._nodes[signal]
    
    # Build result
    result = {'modified': True}
    if updated_array is not None:
        result['updated_element_array'] = updated_array
    
    return result


def setup_to_centers_and_boundaries(
        setup: Setup,
        size: int
    ) -> tuple[dict, dict]:
    """
    Convert a Setup object (e.g., from sparse_uifo) to centers and boundaries dicts.
    
    This function extracts the component types from a setup and creates dictionaries
    in the format expected by the uifo() function. Missing elements are represented
    as "nothing" components.
    
    Parameters
    ----------
    setup : Setup
        The setup object to convert
    size : int
        The grid size (e.g., 3 for a 3x3 grid)
        
    Returns
    -------
    centers : dict
        Dictionary with keys like "11", "23" containing tuples of 
        (component_type, orientation) where component_type is one of 
        ["beamsplitter", "directional_beamsplitter", "nothing"] and 
        orientation is one of ["left", "top", "right", "bottom"]
    boundaries : dict
        Dictionary with keys like "01", "34" containing component_type
        strings, one of ["laser", "squeezer", "detector", "balanced_homodyne", "nothing"]
        
    Examples
    --------
    >>> element_array = jnp.array([...])  # Define sparse UIFO elements
    >>> setup, _ = sparse_uifo(element_array, size=3)
    >>> centers, boundaries = setup_to_centers_and_boundaries(setup, size=3)
    >>> # Now centers and boundaries can be used with uifo()
    >>> setup_dense, _ = uifo(size=3, centers=centers, boundaries=boundaries)
    """
    import re
    
    # Beamsplitter orientation mapping (reverse of what's used in sparse_uifo)
    # Maps element IDs to orientations
    bs_orientation_mapping = {
        6: "left", 7: "top", 8: "left", 9: "top",
        10: "left", 11: "top", 12: "right", 13: "bottom",
        14: "left", 15: "top", 16: "right", 17: "bottom"
    }
    
    # Initialize dictionaries
    centers = {}
    boundaries = {}
    
    # Extract center beamsplitters
    # Pattern: center{x}{y} where x, y are grid coordinates (1 to size)
    for x in range(1, size + 1):
        for y in range(1, size + 1):
            center_name = f"center{x}{y}"
            key = f"{x}{y}"
            
            if center_name in setup._nodes:
                node_info = setup._nodes[center_name]
                component_type = node_info['component']
                
                # Determine orientation by checking which mirrors are connected
                # and which ports they use
                # Default to "left" if we can't determine
                orientation = "left"
                
                # Look for connections to cell mirrors to infer orientation
                mirror_connections = {}
                for (src, tgt), edge_data in setup._edges.items():
                    if src == center_name:
                        # Check if target is a cell mirror
                        if tgt.startswith('ml') or tgt.startswith('mt') or \
                           tgt.startswith('mr') or tgt.startswith('mb'):
                            port = edge_data.get('source_port', 'left')
                            mirror_connections[tgt[:2]] = port
                
                # Infer orientation from mirror connections
                # The orientation tells us which port is the "left" port conceptually
                if 'ml' in mirror_connections:
                    port = mirror_connections['ml']
                    if port == 'left':
                        orientation = 'left'
                    elif port == 'top':
                        orientation = 'top'
                    elif port == 'right':
                        orientation = 'right'
                    elif port == 'bottom':
                        orientation = 'bottom'
                
                centers[key] = (component_type, orientation)
            else:
                # No beamsplitter at this center - use "nothing"
                centers[key] = ("nothing", "left")
    
    # Extract boundary components
    # Boundaries are at coordinates (0, y), (x, 0), (size+1, y), (x, size+1)
    # Pattern: boundary{x}{y} for sources, m{x}{y} for mirrors
    
    # Helper function to extract boundary component type
    def get_boundary_component(x, y):
        """Get the component type at boundary position (x, y)."""
        boundary_name = f"boundary{x}{y}"
        mirror_name = f"m{x}{y}"
        
        # Check for various boundary source types
        # The actual source might have a suffix (e.g., boundary01detector, boundary01lo)
        for node_name, node_info in setup._nodes.items():
            if node_name.startswith(boundary_name):
                component = node_info['component']
                if component == 'laser':
                    # Check if it's part of balanced homodyne (has associated bhbs)
                    bhbs_name = f"{boundary_name}bhbs"
                    if bhbs_name in setup._nodes:
                        return "balanced_homodyne"
                    return "laser"
                elif component == 'squeezer':
                    return "squeezer"
                elif component == 'detector':
                    return "detector"
        
        # If no source found, might just have a mirror (which means "nothing" for source)
        if mirror_name in setup._nodes:
            return "nothing"
        
        return "nothing"
    
    # Top boundary: (0, 1) to (0, size)
    for y in range(1, size + 1):
        key = f"0{y}"
        boundaries[key] = get_boundary_component(0, y)
    
    # Right boundary: (1, size+1) to (size, size+1)
    for x in range(1, size + 1):
        key = f"{x}{size+1}"
        boundaries[key] = get_boundary_component(x, size + 1)
    
    # Bottom boundary: (size+1, size) to (size+1, 1)
    for y in range(size, 0, -1):
        key = f"{size+1}{y}"
        boundaries[key] = get_boundary_component(size + 1, y)
    
    # Left boundary: (size, 0) to (1, 0)
    for x in range(size, 0, -1):
        key = f"{x}0"
        boundaries[key] = get_boundary_component(x, 0)
    
    return centers, boundaries

#-------------------------------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def _map_nodes_to_element_positions(nodes_to_remove: set, n: int) -> list:
    """
    Map node names to their positions in the element array.
    
    Parameters
    ----------
    nodes_to_remove : set
        Set of node names to remove
    n : int
        Grid size
        
    Returns
    -------
    list
        List of positions in the element array to zero out
    """
    import re
    
    positions = []
    
    # Prepare boundary info for mapping
    boundary_info = _prepare_boundary_info(n)
    
    for node_name in nodes_to_remove:
        # Parse node name to determine its type and position
        
        # Boundary sources: boundaryXY (where X,Y are grid coordinates)
        if node_name.startswith('boundary') and not any(suffix in node_name for suffix in ['detector', 'lo', 'bhbs', 'noise']):
            match = re.match(r'boundary(\d+)(\d+)', node_name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Find which boundary index this corresponds to
                for boundary_idx, (side, pos_idx, bx, by, is_row, idx_in_row_col) in enumerate(boundary_info):
                    if bx == x and by == y:
                        positions.append(boundary_idx)
                        break
        
        # Boundary mirrors: mXY
        elif node_name.startswith('m') and len(node_name) >= 3 and node_name[1].isdigit():
            match = re.match(r'm(\d+)(\d+)', node_name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Find which boundary index this corresponds to
                for boundary_idx, (side, pos_idx, bx, by, is_row, idx_in_row_col) in enumerate(boundary_info):
                    if bx == x and by == y:
                        # Boundary mirror position starts after sources
                        positions.append(n * 4 + boundary_idx)
                        break
        
        # Cell mirrors: mlXY, mtXY, mrXY, mbXY (where X,Y are 1-indexed cell coordinates)
        elif node_name.startswith('ml') or node_name.startswith('mt') or \
             node_name.startswith('mr') or node_name.startswith('mb'):
            mirror_type = node_name[:2]  # ml, mt, mr, or mb
            match = re.match(r'[a-z]{2}(\d+)(\d+)', node_name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Convert to 0-indexed
                row_idx, col_idx = x - 1, y - 1
                
                if 0 <= row_idx < n and 0 <= col_idx < n:
                    # Determine which of the 4 mirrors this is
                    mirror_idx = {'ml': 0, 'mt': 1, 'mr': 2, 'mb': 3}.get(mirror_type, 0)
                    
                    # Cell mirrors start at position 8*n
                    # Order: row_idx, col_idx, mirror_idx
                    flat_idx = row_idx * n * 4 + col_idx * 4 + mirror_idx
                    positions.append(8 * n + flat_idx)
        
        # Cell beamsplitters: centerXY (where X,Y are 1-indexed cell coordinates)
        elif node_name.startswith('center'):
            match = re.match(r'center(\d+)(\d+)', node_name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Convert to 0-indexed
                row_idx, col_idx = x - 1, y - 1
                
                if 0 <= row_idx < n and 0 <= col_idx < n:
                    # Beamsplitters start at position 8*n + 4*n*n
                    flat_idx = row_idx * n + col_idx
                    positions.append(8 * n + 4 * n * n + flat_idx)
    
    return positions


def _extract_elements(element_array: jnp.ndarray, n: int) -> dict:
    """
    Extract and organize elements from the element array.
    
    Parameters
    ----------
    element_array : jnp.ndarray
        The flat element array
    n : int
        Grid size
        
    Returns
    -------
    dict
        Dictionary containing organized element arrays
    """
    n_sources = n * 4
    n_boundary_mirrors = 4 * n
    n_cell_mirrors = 4 * n * n
    n_beamsplitters = n * n
    
    # Separate elements into groups
    source_elements = element_array[:n_sources]
    boundary_mirrors = element_array[n_sources:n_sources + n_boundary_mirrors]
    cell_mirrors = element_array[n_sources + n_boundary_mirrors:n_sources + n_boundary_mirrors + n_cell_mirrors].reshape(n, n, 4)
    beamsplitters = element_array[n_sources + n_boundary_mirrors + n_cell_mirrors:n_sources + n_boundary_mirrors + n_cell_mirrors + n_beamsplitters].reshape(n, n)
    
    return {
        'source_elements': source_elements,
        'boundary_mirrors': boundary_mirrors,
        'cell_mirrors': cell_mirrors,
        'beamsplitters': beamsplitters
    }


def _connect_source_to_mirror(S: Setup, source_type: str, source_name: str, mirror_name: str):
    """
    Connect a source (laser, squeezer, or detector) to a boundary mirror.
    
    Parameters
    ----------
    S : Setup
        The setup object
    source_type : str
        Type of source: "detector", "laser", or "squeezer"
    source_name : str
        Name of the source element
    mirror_name : str
        Name of the mirror element
    """
    if source_type == "detector":
        # Detectors are connected directly via target parameter
        S.add("detector", f"{source_name}detector", target=mirror_name, port="left", direction="out")
        S.add("qnoised", f"{source_name}noise", target=mirror_name, port="left", direction="out")
    else:
        # Lasers and squeezers are connected via space
        S.space(source_name, mirror_name, length=1.0, target_port="left")


def _prepare_boundary_info(n: int) -> list[tuple[str, int, int, int, bool, int]]:
    """
    Prepare boundary information for the UIFO.
    
    Parameters
    ----------
    n : int
        Grid size

    Returns
    -------
    list[tuple[str, int, int, int, bool, int]]
        Boundary information: (side, pos, x, y, is_row, idx_in_row_col)
        - side: side of the boundary (top, right, bottom, left)
        - pos: position on the boundary (0 to n-1)
        - x: x-coordinate of the boundary
        - y: y-coordinate of the boundary
        - is_row: True if the boundary is a row, False if it is a column
        - idx_in_row_col: index in the row or column
    """
    boundary_info = []  # Store (side, pos, x, y, is_row, idx_in_row_col)
    
    # Top boundary (side 0): positions (0, 1) to (0, n) - these are columns
    for pos_idx in range(n):
        boundary_info.append(("top", pos_idx, 0, pos_idx + 1, False, pos_idx))
    
    # Right boundary (side 1): positions (1, n+1) to (n, n+1) - these are rows
    for pos_idx in range(n):
        boundary_info.append(("right", pos_idx, pos_idx + 1, n + 1, True, pos_idx))
    
    # Bottom boundary (side 2): positions (n+1, n) to (n+1, 1) - these are columns (reversed)
    for pos_idx in range(n):
        boundary_info.append(("bottom", pos_idx, n + 1, n - pos_idx, False, n - pos_idx - 1))
    
    # Left boundary (side 3): positions (n, 0) to (1, 0) - these are rows (reversed)
    for pos_idx in range(n):
        boundary_info.append(("left", pos_idx, n - pos_idx, 0, True, n - pos_idx - 1))

    return boundary_info


def _add_source_element_default(S: Setup, source_element: int, source_name: str, 
                                element_mapping: dict) -> str:
    """
    Add a source element (laser, squeezer, or detector) to the setup with default parameters.
    
    Parameters
    ----------
    S : Setup
        The setup object
    source_element : int
        Element type ID
    source_name : str
        Name for the source
    element_mapping : dict
        Mapping from element IDs to element types
        
    Returns
    -------
    str or None
        Type of source added (or None if no source)
    """
    source_type = element_mapping[int(source_element)]
    
    if source_type == "detector":
        # Detector has no parameters
        return "detector"
    elif source_type == "laser":
        # Use default laser parameters from DEFAULT_PROPERTIES
        S.add("laser", source_name)
        return "laser"
    elif source_type == "squeezer":
        # Use default squeezer parameters from DEFAULT_PROPERTIES
        S.add("squeezer", source_name)
        return "squeezer"
    
    return None


def _add_mirror_element_default(S: Setup, mirror_element: int, mirror_name: str):
    """
    Add a mirror element to the setup with default parameters.
    
    Parameters
    ----------
    S : Setup
        The setup object
    mirror_element : int
        Element type ID (4 or 5)
    mirror_name : str
        Name for the mirror
    """
    # Use default mirror parameters from DEFAULT_PROPERTIES
    S.add("mirror", mirror_name)
    
    # Add free mass if element type is 5 (mirror with free mass)
    if int(mirror_element) == 5:
        S.add("free_mass", f"{mirror_name}sus", target=mirror_name)


def _add_beamsplitter_element_default(S: Setup, bs_element: int, bs_name: str,
                                      element_mapping: dict, bs_orientation_mapping: dict) -> tuple:
    """
    Add a beamsplitter element to the setup with default parameters.
    
    Parameters
    ----------
    S : Setup
        The setup object
    bs_element : int
        Element type ID (6-17)
    bs_name : str
        Name for the beamsplitter
    element_mapping : dict
        Mapping from element IDs to element types
    bs_orientation_mapping : dict
        Mapping from element IDs to port orientations
        
    Returns
    -------
    tuple
        (bs_type, orientation) - type and orientation of the beamsplitter
    """
    bs_type = element_mapping[bs_element]
    orientation = bs_orientation_mapping[bs_element]
    
    # Add beamsplitter to setup with default parameters
    if bs_type == "beamsplitter":
        S.add("beamsplitter", bs_name)
    else:  # directional_beamsplitter
        S.add("directional_beamsplitter", bs_name)
    
    # Add free mass if applicable (element IDs 8, 9, 14-17 have free mass)
    if bs_element in (8, 9, 14, 15, 16, 17):
        S.add("free_mass", f"{bs_name}sus", target=bs_name)
    
    return (bs_type, orientation)
