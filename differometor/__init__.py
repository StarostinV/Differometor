from differometor.setups import Setup, get_default_parameter_array, uifo, setup_to_centers_and_boundaries
from differometor.sparse_setups import (
    sparse_uifo, 
    build_grid_parameter_converter, 
    grid_params_dict_to_array,
    prune_disconnected_components,
    setup_to_centers_and_boundaries as sparse_setup_to_centers_and_boundaries
)
from differometor.simulate import run, run_build_step, run_simulation_step, simulate_in_parallel
from differometor.plot import plot_uifo_setup