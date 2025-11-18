import os
import re
import numpy as np
import matplotlib.pyplot as plt
from differometor.components import HARD_SIDE_POWER_THRESHOLD, SOFT_SIDE_POWER_THRESHOLD, DETECTOR_POWER_THRESHOLD

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_comparison(
        x, 
        y_1, 
        y_2=None, 
        plot_directory=".", 
        name="sensitivity", 
        y_1_name='optimization', 
        y_2_name='baseline'
    ):
    os.makedirs(plot_directory, exist_ok=True)

    plt.figure()
    if y_2 is not None:
        plt.plot(x, y_2, label=y_2_name, linestyle='--', marker='o', markersize=2, linewidth=3)
    plt.plot(x, y_1, label=y_1_name, linestyle='-', marker='o', markersize=2, linewidth=1)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain Sensitivity [1/sqrt(Hz)]")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/{name}.png")
    plt.close()


def plot_best_losses(
        best_losses, 
        folder, 
        power_violations, 
        suffix=""
    ):
    best_losses = np.asarray(best_losses, dtype=float)
    power_violations = np.asarray(power_violations, dtype=bool)
    n = best_losses.shape[0]
    idx = np.arange(n)
    viol_mask = power_violations
    ok_mask = ~power_violations

    arg_min = int(np.nanargmin(best_losses))
    min_val = float(best_losses[arg_min])
    mean_val = float(np.nanmean(best_losses))

    plt.figure()
    plt.scatter(idx[viol_mask], best_losses[viol_mask], marker='x', color="blue", label="Violating")
    plt.scatter(idx[ok_mask], best_losses[ok_mask], marker='o', color="blue", label="Non violating")
    min_marker = 'x' if power_violations[arg_min] else 'o'
    plt.scatter([arg_min], [min_val], color='r', marker=min_marker, label=f"Best: {arg_min}")
    plt.axhline(y=mean_val, linestyle='--', color='black', label='Mean Losses')

    plt.ylabel('best loss')
    plt.xlabel('run')
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=0 if min_val > 0 else min_val * 1.1)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.savefig(f"{folder}/best_losses{suffix}.png")
    plt.close()


def plot_loss_curve(
        losses, 
        folder
    ):
    os.makedirs(folder, exist_ok=True)

    losses = np.asarray(losses, dtype=float)
    x_full = np.arange(len(losses))

    # Keep only finite values for plotting
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        return
    losses = losses[finite_mask]
    x_full = x_full[finite_mask]

    def filter_large_jumps(arr, xs):
        arr = np.asarray(arr)
        xs = np.asarray(xs)
        if arr.size == 0:
            return arr, xs
        filtered = [arr[0]]
        filtered_x = [xs[0]]
        for i in range(1, len(arr)):
            if arr[i] <= 2 * filtered[-1]:
                filtered.append(arr[i])
                filtered_x.append(xs[i])
        return np.asarray(filtered), np.asarray(filtered_x)

    for loss_scale in ['lin', 'log']:
        for loss_type in ["", "smoothed"]:
            if loss_type == "smoothed":
                y, x = filter_large_jumps(losses, x_full)
            else:
                y, x = losses, x_full

            if y.size == 0:
                continue

            plt.figure()
            plt.plot(x, y, label=f"{loss_type or 'raw'}")
            plt.ylabel('loss')
            plt.xlabel('iteration')
            plt.legend()
            plt.grid(True)

            if loss_scale == 'log':
                # If any non-positive values exist, use symmetrical log scale.
                if np.any(y <= 0):
                    # Choose a linear region around zero so negatives/zero are visible.
                    # Use the smallest non-zero |y| as a guide.
                    nonzero = np.abs(y[np.nonzero(y)])
                    linthresh = float(np.nanmin(nonzero)) if nonzero.size else 1e-8
                    linthresh = max(linthresh, 1e-8)
                    plt.yscale('symlog', linthresh=linthresh, linscale=1.0, base=10)
                    suffix = "log_symlog"
                else:
                    plt.yscale('log')
                    suffix = "log"
                plt.tight_layout()
                plt.savefig(f"{folder}/{suffix}_losses_{loss_type or 'raw'}.png", dpi=200)
            else:
                plt.tight_layout()
                plt.savefig(f"{folder}/losses_{loss_type or 'raw'}.png", dpi=200)

            plt.close()


def plot_powers(
        hard_side_powers, 
        soft_side_powers, 
        detector_powers, 
        suffix, 
        folder
    ):
    def plot_bar_diagram(powers, cutoff, name):
        plt.figure()
        plt.bar(np.arange(len(powers)), powers.squeeze())
        plt.axhline(y=cutoff, color='r', linestyle='--')
        plt.ylabel('Power [W]')
        plt.xlabel('component')
        plt.yscale('log')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"{folder}/powers_{name}{suffix}.png")
        plt.close()

    plot_bar_diagram(hard_side_powers, HARD_SIDE_POWER_THRESHOLD, "hard_side")
    plot_bar_diagram(soft_side_powers, SOFT_SIDE_POWER_THRESHOLD, "soft_side")
    plot_bar_diagram(detector_powers, DETECTOR_POWER_THRESHOLD, "detector")


def plot_uifo_setup(setup, output_file=None, title="UIFO Setup"):
    """
    Plot a UIFO setup using plotly, placing elements on a grid according to their names.
    
    Features rich interactive visualization with:
    - Color-coded components by type
    - Comprehensive hover information for nodes showing all parameters
    - Automatic free_mass detection for mirrors and beamsplitters
    - Interactive edge hover showing connection details (length, ports)
    - Smart boundary positioning to separate sources from mirrors
    
    Parameters
    ----------
    setup : Setup
        The Setup object containing nodes and edges
    output_file : str, optional
        Path to save the HTML plot. If None, the plot is displayed in browser.
    title : str, optional
        Title for the plot
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
        
    Examples
    --------
    >>> from differometor.setups import uifo
    >>> S, params = uifo(size=3)
    >>> fig = plot_uifo_setup(S, output_file="uifo_layout.html")
    >>> # Hover over any component to see all its parameters
    >>> # Hover over connections to see length and port information
    
    Notes
    -----
    - Requires plotly package: `pip install plotly`
    - Automatically detects and displays mass for components with free_mass
    - Boundary sources are offset from mirrors to prevent overlap
    - Numeric values are formatted intelligently (scientific notation for very small/large)
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for plot_uifo_setup. Install with: pip install plotly")
    
    # Component types to skip
    skip_components = {'frequency', 'signal', 'qnoised', 'qhd', 'free_mass'}
    
    # Color mapping for different component types
    color_map = {
        'laser': '#FF6B6B',          # Red
        'squeezer': '#4ECDC4',       # Cyan
        'detector': '#95E1D3',       # Light cyan
        'mirror': '#FFE66D',         # Yellow
        'beamsplitter': '#A8E6CF',   # Light green
        'directional_beamsplitter': '#FFD3B6',  # Peach
        'nothing': '#DCDCDC',        # Light gray
    }
    
    # Symbol mapping for different component types
    symbol_map = {
        'laser': 'star',
        'squeezer': 'diamond',
        'detector': 'square',
        'mirror': 'circle',
        'beamsplitter': 'hexagon',
        'directional_beamsplitter': 'octagon',
        'nothing': 'x',
    }
    
    def parse_node_name(name):
        """
        Extract x, y coordinates from node names (non-boundary nodes).
        
        Patterns:
        - m{x}{y}: boundary mirrors
        - center{x}{y}: cell beamsplitters
        - ml{x}{y}, mt{x}{y}, mr{x}{y}, mb{x}{y}: cell mirrors (left, top, right, bottom)
        
        Returns (x, y) coordinates or None if not parseable
        """
        # Try mirror pattern: m01, m12, etc.
        match = re.match(r'm(\d+)(\d+)$', name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return (x, y)
        
        # Try center pattern: center11, center23, etc.
        match = re.match(r'center(\d+)(\d+)', name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return (x, y)
        
        # Try cell mirror pattern: ml11, mt23, mr12, mb33, etc.
        match = re.match(r'm[ltrb](\d+)(\d+)', name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            # Offset mirrors slightly from center based on direction
            prefix = name[:2]
            offset = 0.15
            if prefix == 'ml':  # left
                return (x, y - offset)
            elif prefix == 'mt':  # top
                return (x - offset, y)
            elif prefix == 'mr':  # right
                return (x, y + offset)
            elif prefix == 'mb':  # bottom
                return (x + offset, y)
        
        return None
    
    # First pass: determine grid boundaries by examining all nodes
    # This helps us properly offset boundary sources
    boundary_coords = []
    for node_name, node_info in setup.nodes():
        # Look for boundary and mirror nodes to determine grid size
        match = re.match(r'(?:boundary|m)(\d+)(\d+)', node_name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            boundary_coords.append((x, y))
    
    # Determine grid bounds
    if boundary_coords:
        max_x = max(x for x, y in boundary_coords)
        max_y = max(y for x, y in boundary_coords)
        min_x = min(x for x, y in boundary_coords)
        min_y = min(y for x, y in boundary_coords)
    else:
        max_x, max_y, min_x, min_y = 0, 0, 0, 0
    
    def parse_node_name_with_bounds(name):
        """Parse node name with knowledge of grid boundaries."""
        # Try boundary pattern with optional suffix: boundary01, boundary12detector, boundary01lo, etc.
        match = re.match(r'boundary(\d+)(\d+)', name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            source_offset = 0.5  # Extra distance from grid
            
            # Determine which boundary based on coordinates and grid bounds
            if x == min_x:  # Top boundary
                return (x - source_offset, y)
            elif y == min_y:  # Left boundary
                return (x, y - source_offset)
            elif x == max_x:  # Bottom boundary
                return (x + source_offset, y)
            elif y == max_y:  # Right boundary
                return (x, y + source_offset)
            else:
                # Fallback: shouldn't happen for well-formed UIFOs
                return (x, y)
        
        # All other patterns handled by original parse_node_name
        return parse_node_name(name)
    
    # Extract node positions and properties
    node_data = []
    node_positions = {}
    
    for node_name, node_info in setup.nodes():
        component = node_info['component']
        
        # Skip auxiliary components
        if component in skip_components:
            continue
        
        # Parse coordinates with boundary knowledge
        coords = parse_node_name_with_bounds(node_name)
        if coords is None:
            # Skip nodes we can't place
            continue
        
        x, y = coords
        node_positions[node_name] = (x, y)
        
        node_data.append({
            'name': node_name,
            'x': x,
            'y': y,
            'component': component,
            'color': color_map.get(component, '#808080'),
            'symbol': symbol_map.get(component, 'circle'),
        })
    
    # Create edge data
    edge_x = []
    edge_y = []
    edge_midpoints_x = []
    edge_midpoints_y = []
    edge_hover_text = []
    
    for source, target, edge_data in setup.edges():
        if source in node_positions and target in node_positions:
            x0, y0 = node_positions[source]
            x1, y1 = node_positions[target]
            
            # Add edge coordinates (with None to create separate line segments)
            edge_y.extend([x0, x1, None])
            edge_x.extend([y0, y1, None])
            
            # Calculate midpoint for hover interaction
            mid_x = (y0 + y1) / 2
            mid_y = (x0 + x1) / 2
            edge_midpoints_x.append(mid_x)
            edge_midpoints_y.append(mid_y)
            
            # Create detailed hover text for edge
            props = edge_data.get('properties', {})
            hover_parts = [
                f"<b>Connection: {source} → {target}</b>",
                f"Length: {props.get('length', 'N/A'):.4g}" if isinstance(props.get('length'), (int, float)) else f"Length: {props.get('length', 'N/A')}",
            ]
            
            # Add other space properties if present
            if 'refractive_index' in props:
                hover_parts.append(f"Refractive Index: {props['refractive_index']:.4g}")
            
            # Add port information if available
            if 'source_port' in edge_data:
                hover_parts.append(f"Source Port: {edge_data['source_port']}")
            if 'target_port' in edge_data:
                hover_parts.append(f"Target Port: {edge_data['target_port']}")
            
            edge_hover_text.append("<br>".join(hover_parts))
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add edges as lines
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='#2E86AB', width=5),  # Blue color, wider lines
            hoverinfo='skip',
            showlegend=False,
            name='Connections'
        ))
    
    # Add invisible markers at edge midpoints for hover interaction
    if edge_midpoints_x:
        fig.add_trace(go.Scatter(
            x=edge_midpoints_x,
            y=edge_midpoints_y,
            mode='markers',
            marker=dict(size=8, color='rgba(0,0,0,0)', line=dict(width=0)),
            hovertext=edge_hover_text,
            hoverinfo='text',
            showlegend=False,
            name='Edge Info'
        ))
    
    # Group nodes by component type for legend
    component_types = {}
    for node in node_data:
        comp_type = node['component']
        if comp_type not in component_types:
            component_types[comp_type] = []
        component_types[comp_type].append(node)
    
    # Helper function to check for free_mass associated with a component
    def get_free_mass_for_component(component_name):
        """Check if there's a free_mass node connected to this component."""
        mass_name = f"{component_name}sus"
        if mass_name in dict(setup.nodes):
            mass_node = setup.nodes[mass_name]
            if mass_node['component'] == 'free_mass':
                return mass_node.get('properties', {}).get('mass', None)
        return None
    
    # Add nodes by component type
    for comp_type, nodes in component_types.items():
        y_coords = [n['x'] for n in nodes]
        x_coords = [n['y'] for n in nodes]
        names = [n['name'] for n in nodes]
        color = nodes[0]['color']
        symbol = nodes[0]['symbol']
        
        # Create hover text with node details
        hover_text = []
        for node in nodes:
            name = node['name']
            node_info = dict(setup.nodes[name])
            props = node_info.get('properties', {})
            hover_parts = [f"<b>{name}</b>", f"Type: {comp_type}", ""]
            
            # Format numeric values nicely
            def format_value(val):
                if isinstance(val, (int, float)):
                    if abs(val) < 1e-3 or abs(val) > 1e4:
                        return f"{val:.4e}"
                    else:
                        return f"{val:.4g}"
                return val
            
            # Add all properties to hover text based on component type
            if comp_type == 'laser':
                hover_parts.append(f"Power: {format_value(props.get('power', 'N/A'))}")
                hover_parts.append(f"Phase: {format_value(props.get('phase', 'N/A'))}")
            
            elif comp_type == 'squeezer':
                hover_parts.append(f"dB: {format_value(props.get('db', 'N/A'))}")
                hover_parts.append(f"Angle: {format_value(props.get('angle', 'N/A'))}")
            
            elif comp_type == 'mirror':
                hover_parts.append(f"Loss: {format_value(props.get('loss', 'N/A'))}")
                hover_parts.append(f"Reflectivity: {format_value(props.get('reflectivity', 'N/A'))}")
                hover_parts.append(f"Tuning: {format_value(props.get('tuning', 'N/A'))}")
                # Check for free mass
                mass = get_free_mass_for_component(name)
                if mass is not None:
                    hover_parts.append(f"Mass: {format_value(mass)}")
            
            elif comp_type == 'beamsplitter':
                hover_parts.append(f"Loss: {format_value(props.get('loss', 'N/A'))}")
                hover_parts.append(f"Reflectivity: {format_value(props.get('reflectivity', 'N/A'))}")
                hover_parts.append(f"Tuning: {format_value(props.get('tuning', 'N/A'))}")
                hover_parts.append(f"Alpha: {format_value(props.get('alpha', 'N/A'))}")
                # Check for free mass
                mass = get_free_mass_for_component(name)
                if mass is not None:
                    hover_parts.append(f"Mass: {format_value(mass)}")
            
            elif comp_type == 'directional_beamsplitter':
                # Directional beamsplitters typically have no parameters, but check for free mass
                mass = get_free_mass_for_component(name)
                if mass is not None:
                    hover_parts.append(f"Mass: {format_value(mass)}")
                else:
                    hover_parts.append("(No parameters)")
            
            elif comp_type == 'detector':
                # Detectors typically have no parameters
                # Check if there's target/port info
                if 'target' in node_info:
                    hover_parts.append(f"Target: {node_info['target']}")
                if 'port' in node_info:
                    hover_parts.append(f"Port: {node_info['port']}")
                if 'direction' in node_info:
                    hover_parts.append(f"Direction: {node_info['direction']}")
            
            else:
                # For any other component types, show all properties
                if props:
                    for key, val in props.items():
                        hover_parts.append(f"{key.capitalize()}: {format_value(val)}")
                else:
                    hover_parts.append("(No parameters)")
            
            hover_text.append("<br>".join(hover_parts))
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=comp_type,
            marker=dict(
                size=12,
                color=color,
                symbol=symbol,
                line=dict(color='black', width=1)
            ),
            text=names,
            hovertext=hover_text,
            hoverinfo='text',
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Horizontal",
            showgrid=True,
            zeroline=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Vertical",
            showgrid=True,
            zeroline=True,
            gridcolor='lightgray',
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode='closest',
        showlegend=True,
        plot_bgcolor='white',
        width=1000,
        height=1000,
    )

    fig.update_yaxes(autorange="reversed")
    
    # Save or show
    if output_file:
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")
    else:
        fig.show()
    
    return fig
