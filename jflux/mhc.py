"""
Manifold-Constrained Hyper-Connections (MHC) for Flux.

Based on DeepSeek's MHC paper, adapted for diffusion transformers.
"""

import jax.numpy as jnp
from chex import Array
from flax import nnx
from jax.typing import DTypeLike


class SimpleHyperConnection(nnx.Module):
    """
    Minimal MHC: learnable weighted sum of current + previous layer output.
    
    This is the simplest probe to test if cross-layer information helps.
    """
    
    def __init__(
        self,
        hidden_size: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        init_alpha: float = 0.9,
    ):
        self.hidden_size = hidden_size
        # Single learnable scalar - initialized to favor current layer
        self.alpha_logit = nnx.Param(
            jnp.array(self._inverse_sigmoid(init_alpha), dtype=param_dtype)
        )
    
    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse sigmoid to initialize at desired alpha value."""
        return jnp.log(x / (1 - x + 1e-8))
    
    def __call__(self, current: Array, previous: Array | None) -> Array:
        if previous is None:
            return current
        
        alpha = nnx.sigmoid(self.alpha_logit.value)  # constrain to [0,1]
        return alpha * current + (1 - alpha) * previous
    
    def get_alpha(self) -> float:
        """Return the current mixing coefficient."""
        return float(nnx.sigmoid(self.alpha_logit.value))


class HyperConnection(nnx.Module):
    """
    Full Hyper-Connection module with manifold constraint.
    
    Learns to combine outputs from multiple previous layers with 
    dynamic coefficients and manifold projection.
    """
    
    def __init__(
        self,
        hidden_size: int,
        history_len: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        use_manifold_constraint: bool = True,
    ):
        self.hidden_size = hidden_size
        self.history_len = history_len
        self.use_manifold_constraint = use_manifold_constraint
        
        # Learnable mixing coefficients for each history slot
        # +1 for current layer
        self.coeff_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=history_len + 1,  
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        
        # Optional: manifold projection (keeps activations on learned subspace)
        if use_manifold_constraint:
            self.manifold_down = nnx.Linear(
                in_features=hidden_size,
                out_features=hidden_size // 4,
                use_bias=False,
                rngs=rngs,
                param_dtype=param_dtype,
            )
            self.manifold_up = nnx.Linear(
                in_features=hidden_size // 4,
                out_features=hidden_size,
                use_bias=False,
                rngs=rngs,
                param_dtype=param_dtype,
            )
    
    def __call__(
        self, 
        current: Array, 
        history: list[Array],
    ) -> Array:
        """
        Args:
            current: Current layer output [B, L, D]
            history: List of previous layer outputs, most recent first
        
        Returns:
            Combined output with hyper-connections
        """
        # Pad history if needed
        while len(history) < self.history_len:
            history.append(current)  # Use current as fallback
        
        # Truncate if too long
        history = history[:self.history_len]
        
        # Compute dynamic coefficients based on current activation
        # [B, L, history_len+1]
        coeffs = nnx.softmax(self.coeff_proj(current), axis=-1)
        
        # Stack all inputs: [B, L, history_len+1, D]
        all_inputs = jnp.stack([current] + history, axis=-2)
        
        # Weighted combination: [B, L, D]
        combined = jnp.einsum("blhd,blh->bld", all_inputs, coeffs)
        
        # Apply manifold constraint
        if self.use_manifold_constraint:
            # Project to lower dim and back - keeps on learned manifold
            manifold_component = self.manifold_up(self.manifold_down(combined))
            # Residual connection to preserve information
            combined = combined + 0.1 * manifold_component
        
        return combined
    
    def get_mixing_coefficients(self, x: Array) -> Array:
        """Get the mixing coefficients for analysis."""
        return nnx.softmax(self.coeff_proj(x), axis=-1)


class HyperConnectionManager:
    """
    Manages hyper-connections across all layers in the model.
    
    Tracks history and provides connections to each block.
    """
    
    def __init__(
        self,
        num_blocks: int,
        hidden_size: int,
        history_len: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        use_full_mhc: bool = False,
    ):
        self.num_blocks = num_blocks
        self.history_len = history_len
        self.use_full_mhc = use_full_mhc
        
        # Create a connection module for each block
        if use_full_mhc:
            self.connections = [
                HyperConnection(
                    hidden_size=hidden_size,
                    history_len=history_len,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                for _ in range(num_blocks)
            ]
        else:
            self.connections = [
                SimpleHyperConnection(
                    hidden_size=hidden_size,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                for _ in range(num_blocks)
            ]
        
        # History buffer (reset for each forward pass)
        self.history: list[Array] = []
    
    def reset(self):
        """Reset history at the start of each forward pass."""
        self.history = []
    
    def apply(self, block_idx: int, current: Array) -> Array:
        """
        Apply hyper-connection for a specific block.
        
        Args:
            block_idx: Index of the current block
            current: Current block output
        
        Returns:
            Output with hyper-connection applied
        """
        connection = self.connections[block_idx]
        
        if self.use_full_mhc:
            result = connection(current, self.history.copy())
        else:
            previous = self.history[-1] if self.history else None
            result = connection(current, previous)
        
        # Update history
        self.history.append(current)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        
        return result
    
    def get_all_alphas(self) -> list[float]:
        """Get all alpha values for SimpleHyperConnection."""
        if not self.use_full_mhc:
            return [conn.get_alpha() for conn in self.connections]
        return []
    
    def get_trainable_params(self):
        """Return all trainable parameters from MHC modules."""
        params = []
        for conn in self.connections:
            params.extend(nnx.state(conn).flat_state().values())
        return params
