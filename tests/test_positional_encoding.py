"""
Tests for the positional encoding module, specifically prepare_transform_fn.
"""
import pytest
import jax
import jax.numpy as jnp

from differometor.sparse_setups import sparse_uifo
from differometor.positional_encoding import (
    prepare_transform_fn, 
    DEFAULT_BOUNDS,
    RELATIVE_DISTANCE_BOUNDS,
    compute_lengths_batch,
    get_length_indices,
    prepare_batch_indices,
)


class TestPrepareTransformFn:
    """Tests for prepare_transform_fn and restore_params_fn."""
    
    def test_basic_transform_shape(self, sparse_setup):
        """Test that restore_params_fn produces correct output shape."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Generate random scaled params
        key = jax.random.PRNGKey(42)
        scaled_params = jax.random.normal(key, (len(opt_names),))
        
        # Transform
        output = restore_fn(scaled_params)
        
        assert output.shape == (len(params),), \
            f"Expected shape ({len(params)},), got {output.shape}"
    
    def test_zero_scaled_params_gives_midpoint(self, sparse_setup):
        """
        Test that scaled_params = 0 results in the middle of each interval.
        
        sigmoid(0) = 0.5, so each parameter should be at (lower + upper) / 2.
        """
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # All zeros scaled params
        scaled_params = jnp.zeros(len(opt_names))
        output = restore_fn(scaled_params)
        
        # Check specific physical parameters
        for i, (element_name, property_name) in enumerate(params):
            if property_name == 'refractive_index':
                # Fixed parameter - should always be 1.0
                assert jnp.isclose(output[i], 1.0, rtol=1e-5), \
                    f"refractive_index at {i} should be 1.0, got {output[i]}"
            elif property_name == 'length':
                # Length is computed from positional encoding, not a simple midpoint
                # Just check it's positive and within reasonable bounds
                assert output[i] > 0, f"length at {i} should be positive, got {output[i]}"
            else:
                # Physical parameter - should be at midpoint
                bounds = DEFAULT_BOUNDS.get(property_name, (0.0, 1.0))
                expected_midpoint = (bounds[0] + bounds[1]) / 2
                assert jnp.isclose(output[i], expected_midpoint, rtol=0.01), \
                    f"{element_name}_{property_name} at {i}: expected {expected_midpoint}, got {output[i]}"
    
    def test_specific_physical_params_at_midpoint(self, sparse_setup):
        """Test specific physical parameters are at midpoint when scaled=0."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        scaled_params = jnp.zeros(len(opt_names))
        output = restore_fn(scaled_params)
        
        # Find specific parameters to check
        param_checks = {
            'loss': (0.0, 1e-4),
            'reflectivity': (0.0, 1.0),
            'tuning': (-180.0, 180.0),
            'alpha': (0.0, 90.0),
            'mass': (1.0, 100.0),
            'power': (0.0, 10.0),
            'phase': (-180.0, 180.0),
            'db': (0.0, 20.0),
            'angle': (-180.0, 180.0),
        }
        
        found_params = {k: False for k in param_checks}
        
        for i, (element_name, property_name) in enumerate(params):
            if property_name in param_checks:
                bounds = param_checks[property_name]
                expected = (bounds[0] + bounds[1]) / 2
                found_params[property_name] = True
                assert jnp.isclose(output[i], expected, rtol=0.01), \
                    f"{element_name}_{property_name}: expected midpoint {expected}, got {output[i]}"
    
    def test_refractive_index_always_one(self, sparse_setup):
        """Test that refractive_index is always 1.0 regardless of scaled params."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Test with various scaled params
        key = jax.random.PRNGKey(123)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            scaled_params = jax.random.normal(subkey, (len(opt_names),)) * 3  # Large values
            output = restore_fn(scaled_params)
            
            for i, (element_name, property_name) in enumerate(params):
                if property_name == 'refractive_index':
                    assert jnp.isclose(output[i], 1.0, rtol=1e-5), \
                        f"refractive_index at {i} should be 1.0, got {output[i]}"
    
    def test_length_parameter_computation(self, sparse_setup):
        """Test that length parameters are computed correctly from positional encoding."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Find a length parameter
        length_indices = []
        length_connections = []
        for i, (element_name, property_name) in enumerate(params):
            if property_name == 'length':
                length_indices.append(i)
                length_connections.append(element_name)
        
        assert len(length_indices) > 0, "No length parameters found in params"
        
        # Test with zero scaled params (sigmoid(0) = 0.5 for all positional)
        scaled_params = jnp.zeros(len(opt_names))
        output = restore_fn(scaled_params)
        
        # Verify lengths are positive and reasonable
        for idx, conn_name in zip(length_indices, length_connections):
            length_value = output[idx]
            assert length_value > 0, f"Length {conn_name} should be positive, got {length_value}"
            # With sigmoid(0)=0.5, lengths should be roughly in middle of bounds
            # Default length bounds are (0.1, 4000)
            assert length_value < 10000, f"Length {conn_name} seems too large: {length_value}"
    
    def test_length_matches_direct_computation(self, sparse_setup):
        """Test that length from restore_fn matches direct computation."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Find physical params count (everything before positional encoding)
        n_physical = len([p for p in params if p[1] not in ('length', 'refractive_index')])
        positional_encoding_names = opt_names[n_physical:]
        n_positional = len(positional_encoding_names)
        
        # Find length parameters
        length_indices = []
        length_connections = []
        for i, (element_name, property_name) in enumerate(params):
            if property_name == 'length':
                length_indices.append(i)
                length_connections.append(element_name)
        
        if len(length_connections) == 0:
            pytest.skip("No length parameters in this setup")
        
        # Use specific scaled params
        key = jax.random.PRNGKey(999)
        scaled_params = jax.random.normal(key, (len(opt_names),))
        output = restore_fn(scaled_params)
        
        # Manually compute distance_array from scaled positional params
        scaled_positional = scaled_params[n_physical:n_physical + n_positional]
        
        # Transform positional encoding same way as in restore_fn
        length_lower, length_upper = DEFAULT_BOUNDS['length']
        rel_lower, rel_upper = RELATIVE_DISTANCE_BOUNDS
        
        n_boundary_dist = 4 * n
        n_row_spacing = n - 1
        n_col_spacing = n - 1
        n_boundary_mirror_rel = 4 * n
        
        idx = 0
        boundary_dist = jax.nn.sigmoid(scaled_positional[idx:idx + n_boundary_dist]) * (length_upper - length_lower) + length_lower
        idx += n_boundary_dist
        row_spacing = jax.nn.sigmoid(scaled_positional[idx:idx + n_row_spacing]) * (length_upper - length_lower) + length_lower
        idx += n_row_spacing
        col_spacing = jax.nn.sigmoid(scaled_positional[idx:idx + n_col_spacing]) * (length_upper - length_lower) + length_lower
        idx += n_col_spacing
        boundary_mirror_rel = jax.nn.sigmoid(scaled_positional[idx:idx + n_boundary_mirror_rel]) * (rel_upper - rel_lower) + rel_lower
        idx += n_boundary_mirror_rel
        cell_mirror_rel = jax.nn.sigmoid(scaled_positional[idx:]) * (rel_upper - rel_lower) + rel_lower
        
        distance_array = jnp.concatenate([
            boundary_dist, row_spacing, col_spacing, boundary_mirror_rel, cell_mirror_rel
        ])
        
        # Compute lengths directly
        all_length_indices = [get_length_indices(conn, n) for conn in length_connections]
        batch_indices = prepare_batch_indices(all_length_indices)
        direct_lengths = compute_lengths_batch(distance_array, batch_indices)
        
        # Compare
        for i, (idx, conn_name) in enumerate(zip(length_indices, length_connections)):
            restored_length = output[idx]
            direct_length = direct_lengths[i]
            assert jnp.isclose(restored_length, direct_length, rtol=1e-5), \
                f"Length {conn_name}: restored={restored_length}, direct={direct_length}"
    
    def test_batch_transform_with_vmap(self, sparse_setup):
        """Test that restore_fn works correctly with vmap for batching."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Create batch of scaled params
        batch_size = 10
        key = jax.random.PRNGKey(42)
        batch_scaled = jax.random.normal(key, (batch_size, len(opt_names)))
        
        # Batch transform with vmap
        restore_batch = jax.vmap(restore_fn)
        batch_output = restore_batch(batch_scaled)
        
        assert batch_output.shape == (batch_size, len(params)), \
            f"Expected shape ({batch_size}, {len(params)}), got {batch_output.shape}"
        
        # Verify each batch element matches individual transform
        for i in range(batch_size):
            individual = restore_fn(batch_scaled[i])
            assert jnp.allclose(batch_output[i], individual, rtol=1e-5), \
                f"Batch element {i} doesn't match individual transform"
    
    def test_jit_compilation(self, sparse_setup):
        """Test that restore_fn can be JIT compiled and produces correct results."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # JIT compile
        restore_jit = jax.jit(restore_fn)
        
        key = jax.random.PRNGKey(42)
        scaled_params = jax.random.normal(key, (len(opt_names),))
        
        # Compare JIT and eager
        eager_output = restore_fn(scaled_params)
        jit_output = restore_jit(scaled_params)
        
        assert jnp.allclose(eager_output, jit_output, rtol=1e-5), \
            "JIT compiled output doesn't match eager output"
    
    def test_differentiability(self, sparse_setup):
        """Test that restore_fn is differentiable."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        def loss_fn(scaled_params):
            output = restore_fn(scaled_params)
            return jnp.sum(output ** 2)
        
        key = jax.random.PRNGKey(42)
        scaled_params = jax.random.normal(key, (len(opt_names),))
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(scaled_params)
        
        assert gradients.shape == scaled_params.shape, \
            f"Gradient shape {gradients.shape} doesn't match input shape {scaled_params.shape}"
        
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(gradients)), "Gradients contain non-finite values"
    
    def test_bounds_respected(self, sparse_setup):
        """Test that transformed parameters respect their bounds."""
        setup, params = sparse_uifo(
            size=sparse_setup['size'],
            element_array=sparse_setup['element_array'],
            mode='space_modulation'
        )
        
        n = sparse_setup['size']
        restore_fn, opt_names = prepare_transform_fn(params, n)
        
        # Test with many random samples
        key = jax.random.PRNGKey(0)
        n_samples = 100
        
        restore_batch = jax.vmap(restore_fn)
        batch_scaled = jax.random.normal(key, (n_samples, len(opt_names)))
        batch_output = restore_batch(batch_scaled)
        
        for i, (element_name, property_name) in enumerate(params):
            col = batch_output[:, i]
            
            if property_name == 'refractive_index':
                # Should always be 1.0
                assert jnp.allclose(col, 1.0, rtol=1e-5), \
                    f"refractive_index should be 1.0"
            elif property_name == 'length':
                # Length should be positive
                assert jnp.all(col > 0), f"Length {element_name} has non-positive values"
            else:
                bounds = DEFAULT_BOUNDS.get(property_name, (0.0, 1.0))
                # Allow small tolerance for numerical precision
                assert jnp.all(col >= bounds[0] - 1e-5), \
                    f"{element_name}_{property_name}: min {col.min()} < lower bound {bounds[0]}"
                assert jnp.all(col <= bounds[1] + 1e-5), \
                    f"{element_name}_{property_name}: max {col.max()} > upper bound {bounds[1]}"

