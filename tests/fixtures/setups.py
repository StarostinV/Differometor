import pytest
import jax.numpy as jnp


def _get_sparse_n2_setup_1():
    """
    Returns a dict defining a sparse setup for size=2 grid.
    
    This function creates the JAX array lazily (when called) to avoid
    backend initialization issues during pytest collection.
    """
    element_array = jnp.array([
        1, 0, 0, 0, 0, 0, 3, 2, # sources
        5, 0, 0, 0, 0, 5, 5, 0, # boundary mirrors
        *([0] * 4), *([0] * 4), 4, 5, 0, 0, *([0] * 4), # cell mirrors
        8, 0, 8, 10
    ])

    elements = [
        ("boundary01detector", "detector"),
        ("boundary20", "squeezer"),
        ("boundary10", "laser"),
        ("m01", "mirror"),
        ("m01sus", "free_mass"),
        ("m01sus", "free_mass"),
        ("ml21", "mirror"),
        ("mt21", "mirror"),
        ("mt21sus", "free_mass"),
        ("center11", "beamsplitter"),
        ("center21", "beamsplitter"),
        ("center22", "directional_beamsplitter"),
    ]

    # tested connections: (source, target, src_port, tgt_port)
    connections = [
        ("m01", "center11", "right", "top"),
        ("mt21", "center21", "left", "top"),
        ("ml21", "center21", "left", "left"),
        ("m20", "ml21", "right", "right"),
        ("boundary20", "m20", "right", "left"),
        ("center21", "center22", "right", "left"),
        ("boundary10", "center11", "right", "left"),
        ("center11", "mt21", "bottom", "right"),
        ("center21", "m31", "bottom", "left"),
    ]

    return {
        'size': 2,
        'element_array': element_array,
        'elements': elements,
        'connections': connections,
    }


# List of setup generator functions (not the setups themselves)
_SPARSE_SETUP_GENERATORS = [_get_sparse_n2_setup_1]


@pytest.fixture(params=_SPARSE_SETUP_GENERATORS)
def sparse_setup(request):
    """
    Parametrized fixture that provides sparse setup configurations.
    
    The fixture calls the generator function to create the setup dict,
    ensuring JAX arrays are created after JAX initialization.
    """
    generator_func = request.param
    return generator_func()
