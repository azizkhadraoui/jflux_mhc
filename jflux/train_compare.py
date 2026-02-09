"""
Comparative training script for MHC experiments on Flux.

Compares three configurations:
1. Baseline Flux (standard residuals)
2. Flux with learned static alphas (from probe)
3. Flux with full MHC (dynamic alphas)

Usage:
    python -m jflux.train_compare --steps 1000
    python -m jflux.train_compare --steps 5000 --batch_size 4
"""

import time
from dataclasses import dataclass
from typing import Literal

import fire
import jax
import jax.numpy as jnp
import optax
import wandb
from chex import Array
from flax import nnx

from jflux.model import Flux, FluxParams
from jflux.modules.layers import timestep_embedding


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Configuration for comparative training."""
    # Model config
    hidden_size: int = 768
    num_heads: int = 12
    depth: int = 6
    depth_single_blocks: int = 6
    mlp_ratio: float = 4.0
    
    # Data config
    batch_size: int = 2
    seq_len_img: int = 256
    seq_len_txt: int = 77
    
    # Training config
    steps: int = 1000
    lr: float = 1e-4
    warmup_steps: int = 100
    log_every: int = 10
    eval_every: int = 100
    
    # MHC config
    mhc_history_len: int = 2
    
    # W&B config
    wandb_project: str = "mhc-flux-compare"
    wandb_entity: str | None = None


# Learned static alphas from probe experiment (500 steps)
LEARNED_STATIC_ALPHAS = [
    0.9000,  # Block 0 (double #0)
    0.9693,  # Block 1 (double #1)
    0.9724,  # Block 2 (double #2)
    0.0371,  # Block 3 (double #3) - SKIP BLOCK
    0.1295,  # Block 4 (double #4) - SKIP BLOCK
    0.9868,  # Block 5 (double #5)
    0.9000,  # Block 6 (single #0)
    0.9644,  # Block 7 (single #1)
    0.0201,  # Block 8 (single #2) - SKIP BLOCK
    0.0158,  # Block 9 (single #3) - SKIP BLOCK
    0.9954,  # Block 10 (single #4)
    0.9907,  # Block 11 (single #5)
]


def get_flux_params(config: TrainConfig, rngs: nnx.Rngs) -> FluxParams:
    """Create params for Flux model."""
    pe_dim = config.hidden_size // config.num_heads
    return FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=config.hidden_size,
        mlp_ratio=config.mlp_ratio,
        num_heads=config.num_heads,
        depth=config.depth,
        depth_single_blocks=config.depth_single_blocks,
        axes_dim=[16, 24, 24],  # sum = 64 = pe_dim for 768/12
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
        rngs=rngs,
        param_dtype=jnp.bfloat16,
    )


def create_batch(config: TrainConfig, key: jax.random.PRNGKey) -> dict:
    """Create a training batch with random data."""
    keys = jax.random.split(key, 7)
    
    # Create random "image" latents and "text" embeddings
    img = jax.random.normal(keys[0], (config.batch_size, config.seq_len_img, 64), dtype=jnp.bfloat16)
    img_ids = jax.random.uniform(keys[1], (config.batch_size, config.seq_len_img, 3))
    txt = jax.random.normal(keys[2], (config.batch_size, config.seq_len_txt, 4096), dtype=jnp.bfloat16)
    txt_ids = jax.random.uniform(keys[3], (config.batch_size, config.seq_len_txt, 3))
    timesteps = jax.random.uniform(keys[4], (config.batch_size,))
    y = jax.random.normal(keys[5], (config.batch_size, 768), dtype=jnp.bfloat16)
    
    # Create random target for loss computation
    target = jax.random.normal(keys[6], (config.batch_size, config.seq_len_img, 64), dtype=jnp.bfloat16)
    
    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "timesteps": timesteps,
        "y": y,
        "target": target,
    }


# ============================================================================
# Model Variants
# ============================================================================

class BaselineFlux:
    """Standard Flux with normal residual connections."""
    
    def __init__(self, config: TrainConfig, rngs: nnx.Rngs):
        self.config = config
        params = get_flux_params(config, rngs)
        self.model = Flux(params)
        self.name = "baseline"
    
    def __call__(self, batch: dict) -> Array:
        return self.model(
            img=batch["img"],
            img_ids=batch["img_ids"],
            txt=batch["txt"],
            txt_ids=batch["txt_ids"],
            timesteps=batch["timesteps"],
            y=batch["y"],
        )
    
    def get_params(self):
        return nnx.state(self.model)


class StaticAlphaFlux:
    """Flux with static learned alpha values from probe."""
    
    def __init__(self, config: TrainConfig, rngs: nnx.Rngs):
        self.config = config
        params = get_flux_params(config, rngs)
        self.model = Flux(params)
        self.name = "static_alpha"
        
        # Use learned static alphas
        self.alphas = jnp.array(LEARNED_STATIC_ALPHAS[:config.depth + config.depth_single_blocks])
    
    def __call__(self, batch: dict) -> Array:
        """Forward pass with static alpha hyper-connections."""
        # Process inputs
        img = self.model.img_in(batch["img"])
        txt = self.model.txt_in(batch["txt"])
        vec = self.model.time_in(timestep_embedding(batch["timesteps"], 256))
        vec = vec + self.model.vector_in(batch["y"])
        
        ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
        pe = self.model.pe_embedder(ids)
        
        # Double blocks with static alpha
        prev_img = None
        block_idx = 0
        for block in self.model.double_blocks.layers:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if prev_img is not None:
                alpha = self.alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_img
            prev_img = img
            block_idx += 1
        
        # Single blocks with static alpha
        img = jnp.concatenate((txt, img), axis=1)
        prev_single = None
        for block in self.model.single_blocks.layers:
            img = block(img, vec=vec, pe=pe)
            if prev_single is not None:
                alpha = self.alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_single
            prev_single = img
            block_idx += 1
        
        img = img[:, batch["txt"].shape[1]:, ...]
        img = self.model.final_layer(img, vec)
        return img
    
    def get_params(self):
        return nnx.state(self.model)


class DynamicMHCFlux:
    """Flux with dynamic MHC (learnable per-layer alphas)."""
    
    def __init__(self, config: TrainConfig, rngs: nnx.Rngs):
        self.config = config
        params = get_flux_params(config, rngs)
        self.model = Flux(params)
        self.name = "dynamic_mhc"
        
        # Initialize alpha logits at 0.9 (logit ≈ 2.2)
        total_blocks = config.depth + config.depth_single_blocks
        init_logit = float(jnp.log(jnp.array(0.9) / (1 - 0.9)))
        self.alpha_logits = jnp.ones(total_blocks) * init_logit
    
    def __call__(self, batch: dict, alpha_logits: Array = None) -> Array:
        """Forward pass with dynamic alpha hyper-connections."""
        if alpha_logits is None:
            alpha_logits = self.alpha_logits
        alphas = nnx.sigmoid(alpha_logits)
        
        # Process inputs
        img = self.model.img_in(batch["img"])
        txt = self.model.txt_in(batch["txt"])
        vec = self.model.time_in(timestep_embedding(batch["timesteps"], 256))
        vec = vec + self.model.vector_in(batch["y"])
        
        ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
        pe = self.model.pe_embedder(ids)
        
        # Double blocks with dynamic alpha
        prev_img = None
        block_idx = 0
        for block in self.model.double_blocks.layers:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if prev_img is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_img
            prev_img = img
            block_idx += 1
        
        # Single blocks with dynamic alpha
        img = jnp.concatenate((txt, img), axis=1)
        prev_single = None
        for block in self.model.single_blocks.layers:
            img = block(img, vec=vec, pe=pe)
            if prev_single is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_single
            prev_single = img
            block_idx += 1
        
        img = img[:, batch["txt"].shape[1]:, ...]
        img = self.model.final_layer(img, vec)
        return img
    
    def get_params(self):
        return nnx.state(self.model), self.alpha_logits
    
    def get_alphas(self) -> list[float]:
        return [float(a) for a in nnx.sigmoid(self.alpha_logits)]


# ============================================================================
# Training Functions
# ============================================================================

def compute_loss(output: Array, target: Array) -> Array:
    """Simple MSE loss for flow matching."""
    return jnp.mean((output - target) ** 2)


def train_baseline(config: TrainConfig, key: jax.random.PRNGKey) -> dict:
    """Train baseline Flux model."""
    print("\n" + "=" * 60)
    print("Training: BASELINE FLUX")
    print("=" * 60)
    
    rngs = nnx.Rngs(default=42)
    model = BaselineFlux(config, rngs)
    
    # Setup optimizer with nnx.Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
    )
    optimizer = nnx.Optimizer(model.model, optax.adam(schedule), wrt=nnx.Param)
    
    metrics = {"loss": [], "step_time": []}
    
    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            output = model(
                img=batch["img"],
                img_ids=batch["img_ids"],
                txt=batch["txt"],
                txt_ids=batch["txt_ids"],
                timesteps=batch["timesteps"],
                y=batch["y"],
            )
            return compute_loss(output, batch["target"])
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss
    
    for step in range(config.steps):
        key, batch_key = jax.random.split(key)
        batch = create_batch(config, batch_key)
        
        start_time = time.time()
        loss = train_step(model.model, optimizer, batch)
        step_time = time.time() - start_time
        
        metrics["loss"].append(float(loss))
        metrics["step_time"].append(step_time)
        
        if step % config.log_every == 0:
            print(f"Step {step:5d} | Loss: {float(loss):.6f} | Time: {step_time:.3f}s")
            wandb.log({
                "baseline/loss": float(loss),
                "baseline/step_time": step_time,
                "step": step,
            })
    
    return metrics


def train_static_alpha(config: TrainConfig, key: jax.random.PRNGKey) -> dict:
    """Train Flux with static learned alphas."""
    print("\n" + "=" * 60)
    print("Training: STATIC ALPHA FLUX")
    print("=" * 60)
    print(f"Using pre-learned alphas: {LEARNED_STATIC_ALPHAS[:config.depth + config.depth_single_blocks]}")
    
    rngs = nnx.Rngs(default=42)
    wrapper = StaticAlphaFlux(config, rngs)
    
    # Setup optimizer with nnx.Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
    )
    optimizer = nnx.Optimizer(wrapper.model, optax.adam(schedule), wrt=nnx.Param)
    
    metrics = {"loss": [], "step_time": []}
    alphas = wrapper.alphas
    
    def forward_with_static_alpha(model, batch, alphas):
        """Forward pass with static alpha hyper-connections."""
        # Process inputs
        img = model.img_in(batch["img"])
        txt = model.txt_in(batch["txt"])
        vec = model.time_in(timestep_embedding(batch["timesteps"], 256))
        vec = vec + model.vector_in(batch["y"])
        
        ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
        pe = model.pe_embedder(ids)
        
        # Double blocks with static alpha
        prev_img = None
        block_idx = 0
        for block in model.double_blocks.layers:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if prev_img is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_img
            prev_img = img
            block_idx += 1
        
        # Single blocks with static alpha
        img = jnp.concatenate((txt, img), axis=1)
        prev_single = None
        for block in model.single_blocks.layers:
            img = block(img, vec=vec, pe=pe)
            if prev_single is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_single
            prev_single = img
            block_idx += 1
        
        img = img[:, batch["txt"].shape[1]:, ...]
        img = model.final_layer(img, vec)
        return img
    
    @nnx.jit
    def train_step(model, optimizer, batch, alphas):
        def loss_fn(model):
            output = forward_with_static_alpha(model, batch, alphas)
            return compute_loss(output, batch["target"])
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss
    
    for step in range(config.steps):
        key, batch_key = jax.random.split(key)
        batch = create_batch(config, batch_key)
        
        start_time = time.time()
        loss = train_step(wrapper.model, optimizer, batch, alphas)
        step_time = time.time() - start_time
        
        metrics["loss"].append(float(loss))
        metrics["step_time"].append(step_time)
        
        if step % config.log_every == 0:
            print(f"Step {step:5d} | Loss: {float(loss):.6f} | Time: {step_time:.3f}s")
            wandb.log({
                "static_alpha/loss": float(loss),
                "static_alpha/step_time": step_time,
                "step": step,
            })
    
    return metrics


def train_dynamic_mhc(config: TrainConfig, key: jax.random.PRNGKey) -> dict:
    """Train Flux with dynamic MHC."""
    print("\n" + "=" * 60)
    print("Training: DYNAMIC MHC FLUX")
    print("=" * 60)
    
    rngs = nnx.Rngs(default=42)
    wrapper = DynamicMHCFlux(config, rngs)
    
    # Setup optimizer for model params with nnx.Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
    )
    model_optimizer = nnx.Optimizer(wrapper.model, optax.adam(schedule), wrt=nnx.Param)
    
    # Separate optimizer for alpha logits (higher LR)
    alpha_optimizer = optax.adam(0.01)
    alpha_opt_state = alpha_optimizer.init(wrapper.alpha_logits)
    
    metrics = {"loss": [], "step_time": [], "alpha_mean": [], "alpha_std": []}
    
    def forward_with_dynamic_alpha(model, batch, alpha_logits):
        """Forward pass with dynamic alpha hyper-connections."""
        alphas = nnx.sigmoid(alpha_logits)
        
        # Process inputs
        img = model.img_in(batch["img"])
        txt = model.txt_in(batch["txt"])
        vec = model.time_in(timestep_embedding(batch["timesteps"], 256))
        vec = vec + model.vector_in(batch["y"])
        
        ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
        pe = model.pe_embedder(ids)
        
        # Double blocks with dynamic alpha
        prev_img = None
        block_idx = 0
        for block in model.double_blocks.layers:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if prev_img is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_img
            prev_img = img
            block_idx += 1
        
        # Single blocks with dynamic alpha
        img = jnp.concatenate((txt, img), axis=1)
        prev_single = None
        for block in model.single_blocks.layers:
            img = block(img, vec=vec, pe=pe)
            if prev_single is not None:
                alpha = alphas[block_idx]
                img = alpha * img + (1 - alpha) * prev_single
            prev_single = img
            block_idx += 1
        
        img = img[:, batch["txt"].shape[1]:, ...]
        img = model.final_layer(img, vec)
        return img
    
    # Compute gradients separately for model and alphas
    def compute_alpha_grads(model, batch, alpha_logits):
        """Compute gradients w.r.t. alpha_logits only."""
        def alpha_loss_fn(alpha_logits):
            output = forward_with_dynamic_alpha(model, batch, alpha_logits)
            return compute_loss(output, batch["target"])
        return jax.value_and_grad(alpha_loss_fn)(alpha_logits)
    
    @nnx.jit
    def train_model_step(model, model_optimizer, batch, alpha_logits):
        """Train step for model parameters only."""
        def loss_fn(model):
            output = forward_with_dynamic_alpha(model, batch, alpha_logits)
            return compute_loss(output, batch["target"])
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        model_optimizer.update(grads)
        return loss
    
    for step in range(config.steps):
        key, batch_key = jax.random.split(key)
        batch = create_batch(config, batch_key)
        
        start_time = time.time()
        
        # Update model params
        loss = train_model_step(wrapper.model, model_optimizer, batch, wrapper.alpha_logits)
        
        # Update alpha logits separately
        _, alpha_grads = compute_alpha_grads(wrapper.model, batch, wrapper.alpha_logits)
        alpha_updates, alpha_opt_state = alpha_optimizer.update(
            alpha_grads, alpha_opt_state, wrapper.alpha_logits
        )
        wrapper.alpha_logits = optax.apply_updates(wrapper.alpha_logits, alpha_updates)
        
        step_time = time.time() - start_time
        
        current_alphas = wrapper.get_alphas()
        metrics["loss"].append(float(loss))
        metrics["step_time"].append(step_time)
        metrics["alpha_mean"].append(sum(current_alphas) / len(current_alphas))
        metrics["alpha_std"].append(float(jnp.std(jnp.array(current_alphas))))
        
        if step % config.log_every == 0:
            alpha_mean = sum(current_alphas) / len(current_alphas)
            alpha_min = min(current_alphas)
            alpha_max = max(current_alphas)
            print(f"Step {step:5d} | Loss: {float(loss):.6f} | Time: {step_time:.3f}s | "
                  f"α: {alpha_mean:.3f} [{alpha_min:.3f}, {alpha_max:.3f}]")
            wandb.log({
                "dynamic_mhc/loss": float(loss),
                "dynamic_mhc/step_time": step_time,
                "dynamic_mhc/alpha_mean": alpha_mean,
                "dynamic_mhc/alpha_min": alpha_min,
                "dynamic_mhc/alpha_max": alpha_max,
                "dynamic_mhc/alpha_std": float(jnp.std(jnp.array(current_alphas))),
                "step": step,
            })
        
        # Log individual alphas periodically
        if step % config.eval_every == 0:
            for i, alpha in enumerate(current_alphas):
                wandb.log({f"dynamic_mhc/alpha_{i}": alpha, "step": step})
    
    # Log final alphas
    final_alphas = wrapper.get_alphas()
    print("\nFinal learned alphas:")
    for i, alpha in enumerate(final_alphas):
        block_type = "double" if i < config.depth else "single"
        local_idx = i if i < config.depth else i - config.depth
        print(f"  Block {i:2d} ({block_type:6s} #{local_idx:2d}): α = {alpha:.4f}")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main(
    steps: int = 1000,
    batch_size: int = 2,
    lr: float = 1e-4,
    hidden_size: int = 768,
    depth: int = 6,
    depth_single_blocks: int = 6,
    log_every: int = 10,
    eval_every: int = 100,
    wandb_project: str = "mhc-flux-compare",
    run_baseline: bool = True,
    run_static: bool = True,
    run_dynamic: bool = True,
):
    """
    Run comparative training experiments.
    
    Args:
        steps: Number of training steps per model
        batch_size: Batch size for training
        lr: Learning rate
        hidden_size: Model hidden dimension
        depth: Number of double blocks
        depth_single_blocks: Number of single blocks
        log_every: Log metrics every N steps
        eval_every: Detailed eval every N steps
        wandb_project: W&B project name
        run_baseline: Whether to train baseline
        run_static: Whether to train static alpha variant
        run_dynamic: Whether to train dynamic MHC variant
    """
    config = TrainConfig(
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        log_every=log_every,
        eval_every=eval_every,
        wandb_project=wandb_project,
    )
    
    print("=" * 60)
    print("MHC COMPARATIVE TRAINING")
    print("=" * 60)
    print(f"Config: {config}")
    print("=" * 60)
    
    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=f"compare-{steps}steps",
        config=vars(config),
    )
    
    # Set up random keys
    key = jax.random.PRNGKey(42)
    
    all_metrics = {}
    
    # Train each variant
    if run_baseline:
        key, subkey = jax.random.split(key)
        all_metrics["baseline"] = train_baseline(config, subkey)
    
    if run_static:
        key, subkey = jax.random.split(key)
        all_metrics["static_alpha"] = train_static_alpha(config, subkey)
    
    if run_dynamic:
        key, subkey = jax.random.split(key)
        all_metrics["dynamic_mhc"] = train_dynamic_mhc(config, subkey)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    summary_table = wandb.Table(columns=["Model", "Final Loss", "Avg Loss (last 100)", "Avg Step Time"])
    
    for name, metrics in all_metrics.items():
        final_loss = metrics["loss"][-1]
        avg_loss_last_100 = sum(metrics["loss"][-100:]) / min(100, len(metrics["loss"]))
        avg_time = sum(metrics["step_time"]) / len(metrics["step_time"])
        
        print(f"\n{name.upper()}:")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Avg loss (last 100): {avg_loss_last_100:.6f}")
        print(f"  Avg step time: {avg_time:.4f}s")
        
        summary_table.add_data(name, final_loss, avg_loss_last_100, avg_time)
        
        wandb.log({
            f"summary/{name}_final_loss": final_loss,
            f"summary/{name}_avg_loss_last_100": avg_loss_last_100,
            f"summary/{name}_avg_step_time": avg_time,
        })
    
    wandb.log({"summary_table": summary_table})
    
    # Compute relative improvements
    if "baseline" in all_metrics and "static_alpha" in all_metrics:
        baseline_final = all_metrics["baseline"]["loss"][-1]
        static_final = all_metrics["static_alpha"]["loss"][-1]
        improvement = (baseline_final - static_final) / baseline_final * 100
        print(f"\nStatic Alpha vs Baseline: {improvement:+.2f}% loss improvement")
        wandb.log({"summary/static_vs_baseline_improvement": improvement})
    
    if "baseline" in all_metrics and "dynamic_mhc" in all_metrics:
        baseline_final = all_metrics["baseline"]["loss"][-1]
        dynamic_final = all_metrics["dynamic_mhc"]["loss"][-1]
        improvement = (baseline_final - dynamic_final) / baseline_final * 100
        print(f"Dynamic MHC vs Baseline: {improvement:+.2f}% loss improvement")
        wandb.log({"summary/dynamic_vs_baseline_improvement": improvement})
    
    wandb.finish()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return all_metrics


if __name__ == "__main__":
    fire.Fire(main)
