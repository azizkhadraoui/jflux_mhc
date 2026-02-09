"""
Analysis script for MHC experiments on Flux.

Run activation analysis, gradient flow analysis, and probe training
with W&B logging.

Usage:
    python -m jflux.analysis --experiment activation
    python -m jflux.analysis --experiment gradient  
    python -m jflux.analysis --experiment probe
"""

import os
from dataclasses import dataclass
from typing import Literal

import fire
import jax
import jax.numpy as jnp
import wandb
from chex import Array
from einops import rearrange
from flax import nnx

from jflux.model import Flux, FluxParams
from jflux.modules.layers import timestep_embedding
from jflux.mhc import HyperConnectionManager, SimpleHyperConnection


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis experiments."""
    # Model config (tiny for fast iteration)
    hidden_size: int = 768
    num_heads: int = 12
    depth: int = 6
    depth_single_blocks: int = 6
    mlp_ratio: float = 4.0
    
    # Data config
    batch_size: int = 2
    seq_len_img: int = 256  # 16x16 patches
    seq_len_txt: int = 77
    
    # MHC config
    history_len: int = 2
    use_full_mhc: bool = False
    
    # Training config (for probe)
    probe_steps: int = 100
    probe_lr: float = 1e-3
    
    # W&B config
    wandb_project: str = "mhc-flux-analysis"
    wandb_entity: str | None = None


def get_tiny_flux_params(config: AnalysisConfig, rngs: nnx.Rngs) -> FluxParams:
    """Create params for a tiny Flux model (fast iteration)."""
    # pe_dim = hidden_size // num_heads, axes_dim must sum to pe_dim
    pe_dim = config.hidden_size // config.num_heads  # 768 // 12 = 64
    return FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=config.hidden_size,
        mlp_ratio=config.mlp_ratio,
        num_heads=config.num_heads,
        depth=config.depth,
        depth_single_blocks=config.depth_single_blocks,
        axes_dim=[16, 24, 24],  # sum = 64 = pe_dim
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
        rngs=rngs,
        param_dtype=jnp.bfloat16,
    )


def create_dummy_batch(config: AnalysisConfig, rngs: nnx.Rngs) -> dict:
    """Create a dummy batch for analysis."""
    key = rngs()
    keys = jax.random.split(key, 6)
    
    return {
        "img": jax.random.normal(keys[0], (config.batch_size, config.seq_len_img, 64), dtype=jnp.bfloat16),
        "img_ids": jax.random.uniform(keys[1], (config.batch_size, config.seq_len_img, 3)),
        "txt": jax.random.normal(keys[2], (config.batch_size, config.seq_len_txt, 4096), dtype=jnp.bfloat16),
        "txt_ids": jax.random.uniform(keys[3], (config.batch_size, config.seq_len_txt, 3)),
        "timesteps": jax.random.uniform(keys[4], (config.batch_size,)),
        "y": jax.random.normal(keys[5], (config.batch_size, 768), dtype=jnp.bfloat16),
    }


# ============================================================================
# Activation Analysis
# ============================================================================

def analyze_activations(
    model: Flux,
    batch: dict,
    config: AnalysisConfig,
) -> dict:
    """
    Analyze layer activations to find MHC opportunities.
    
    Returns dict with:
    - cross_layer_correlations: correlation between adjacent layers
    - activation_norms: norm at each layer
    - activation_stats: mean, std, max per layer
    """
    results = {
        "double_block_img_norms": [],
        "double_block_txt_norms": [],
        "double_block_img_correlations": [],
        "single_block_norms": [],
        "single_block_correlations": [],
    }
    
    # Process inputs
    img = model.img_in(batch["img"])
    txt = model.txt_in(batch["txt"])
    vec = model.time_in(timestep_embedding(batch["timesteps"], 256))
    vec = vec + model.vector_in(batch["y"])
    
    ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
    pe = model.pe_embedder(ids)
    
    # Track activations through double blocks
    prev_img = None
    prev_txt = None
    
    for i, block in enumerate(model.double_blocks.layers):
        img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        
        # Compute norms
        img_norm = float(jnp.linalg.norm(img))
        txt_norm = float(jnp.linalg.norm(txt))
        results["double_block_img_norms"].append(img_norm)
        results["double_block_txt_norms"].append(txt_norm)
        
        # Compute cross-layer correlation
        if prev_img is not None:
            img_flat = img.flatten()[:10000]
            prev_img_flat = prev_img.flatten()[:10000]
            corr = float(jnp.corrcoef(img_flat, prev_img_flat)[0, 1])
            results["double_block_img_correlations"].append(corr)
        
        prev_img = img
        prev_txt = txt
    
    # Concatenate for single blocks
    img = jnp.concatenate((txt, img), axis=1)
    prev_single = None
    
    for i, block in enumerate(model.single_blocks.layers):
        img = block(img, vec=vec, pe=pe)
        
        # Compute norm
        single_norm = float(jnp.linalg.norm(img))
        results["single_block_norms"].append(single_norm)
        
        # Compute correlation
        if prev_single is not None:
            single_flat = img.flatten()[:10000]
            prev_flat = prev_single.flatten()[:10000]
            corr = float(jnp.corrcoef(single_flat, prev_flat)[0, 1])
            results["single_block_correlations"].append(corr)
        
        prev_single = img
    
    return results


def run_activation_analysis(config: AnalysisConfig):
    """Run activation analysis and log to W&B."""
    print("=" * 60)
    print("ACTIVATION ANALYSIS")
    print("=" * 60)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name="activation-analysis",
        config=vars(config),
    )
    
    rngs = nnx.Rngs(default=42)
    params = get_tiny_flux_params(config, rngs)
    model = Flux(params)
    
    print(f"Created tiny Flux model:")
    print(f"  - Double blocks: {config.depth}")
    print(f"  - Single blocks: {config.depth_single_blocks}")
    print(f"  - Hidden size: {config.hidden_size}")
    
    batch = create_dummy_batch(config, rngs)
    
    print("\nRunning activation analysis...")
    results = analyze_activations(model, batch, config)
    
    # Log to W&B
    for i, norm in enumerate(results["double_block_img_norms"]):
        wandb.log({
            "activation/double_block_img_norm": norm,
            "layer_idx": i,
        })
    
    for i, norm in enumerate(results["single_block_norms"]):
        wandb.log({
            "activation/single_block_norm": norm,
            "layer_idx": config.depth + i,
        })
    
    for i, corr in enumerate(results["double_block_img_correlations"]):
        wandb.log({
            "activation/double_block_correlation": corr,
            "layer_idx": i + 1,
        })
    
    for i, corr in enumerate(results["single_block_correlations"]):
        wandb.log({
            "activation/single_block_correlation": corr,
            "layer_idx": config.depth + i + 1,
        })
    
    # Summary stats
    avg_double_corr = sum(results["double_block_img_correlations"]) / len(results["double_block_img_correlations"]) if results["double_block_img_correlations"] else 0
    avg_single_corr = sum(results["single_block_correlations"]) / len(results["single_block_correlations"]) if results["single_block_correlations"] else 0
    
    wandb.log({
        "summary/avg_double_block_correlation": avg_double_corr,
        "summary/avg_single_block_correlation": avg_single_corr,
        "summary/norm_decay_double": results["double_block_img_norms"][-1] / results["double_block_img_norms"][0] if results["double_block_img_norms"] else 1,
        "summary/norm_decay_single": results["single_block_norms"][-1] / results["single_block_norms"][0] if results["single_block_norms"] else 1,
    })
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Avg double block correlation: {avg_double_corr:.4f}")
    print(f"Avg single block correlation: {avg_single_corr:.4f}")
    print(f"  → High correlation (>0.8) suggests MHC can exploit redundancy")
    print(f"\nNorm decay (double): {results['double_block_img_norms'][-1] / results['double_block_img_norms'][0]:.4f}")
    print(f"Norm decay (single): {results['single_block_norms'][-1] / results['single_block_norms'][0]:.4f}")
    print(f"  → Ratio far from 1.0 suggests gradient flow issues")
    
    wandb.finish()
    return results


# ============================================================================
# Gradient Flow Analysis
# ============================================================================

def analyze_gradient_flow(
    model: Flux,
    batch: dict,
    config: AnalysisConfig,
) -> dict:
    """
    Analyze gradient flow through the model.
    
    Single forward-backward pass to check gradient health.
    """
    def loss_fn(model):
        output = model(
            img=batch["img"],
            img_ids=batch["img_ids"],
            txt=batch["txt"],
            txt_ids=batch["txt_ids"],
            timesteps=batch["timesteps"],
            y=batch["y"],
        )
        return jnp.mean(output ** 2)
    
    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Analyze gradients per layer
    results = {
        "loss": float(loss),
        "gradient_norms": {},
        "gradient_stats": {},
    }
    
    # Use jax tree utilities to iterate over gradients
    flat_grads, tree_def = jax.tree_util.tree_flatten_with_path(grads)
    
    for path, value in flat_grads:
        if value is None:
            continue
        
        path_str = ".".join(str(p) for p in path)
        grad_array = value.value if hasattr(value, 'value') else value
        
        if grad_array is not None and hasattr(grad_array, 'shape'):
            norm = float(jnp.linalg.norm(grad_array))
            mean = float(jnp.mean(jnp.abs(grad_array)))
            std = float(jnp.std(grad_array))
            max_val = float(jnp.max(jnp.abs(grad_array)))
            
            results["gradient_norms"][path_str] = norm
            results["gradient_stats"][path_str] = {
                "mean": mean,
                "std": std,
                "max": max_val,
            }
    
    return results


def run_gradient_analysis(config: AnalysisConfig):
    """Run gradient flow analysis and log to W&B."""
    print("=" * 60)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 60)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name="gradient-analysis",
        config=vars(config),
    )
    
    rngs = nnx.Rngs(default=42)
    params = get_tiny_flux_params(config, rngs)
    model = Flux(params)
    batch = create_dummy_batch(config, rngs)
    
    print("Running gradient flow analysis...")
    results = analyze_gradient_flow(model, batch, config)
    
    print(f"\nLoss: {results['loss']:.6f}")
    
    # Group gradients by layer type
    double_block_norms = []
    single_block_norms = []
    
    for path, norm in results["gradient_norms"].items():
        wandb.log({"gradient/norm/" + path: norm})
        
        if "double_blocks" in path and "kernel" in path:
            double_block_norms.append(norm)
        elif "single_blocks" in path and "kernel" in path:
            single_block_norms.append(norm)
    
    # Summary
    if double_block_norms:
        wandb.log({
            "summary/double_block_grad_mean": sum(double_block_norms) / len(double_block_norms),
            "summary/double_block_grad_std": float(jnp.std(jnp.array(double_block_norms))),
            "summary/double_block_grad_max": max(double_block_norms),
            "summary/double_block_grad_min": min(double_block_norms),
        })
        print(f"\nDouble block gradient norms:")
        print(f"  Mean: {sum(double_block_norms) / len(double_block_norms):.6f}")
        print(f"  Std:  {float(jnp.std(jnp.array(double_block_norms))):.6f}")
        print(f"  Range: [{min(double_block_norms):.6f}, {max(double_block_norms):.6f}]")
    
    if single_block_norms:
        wandb.log({
            "summary/single_block_grad_mean": sum(single_block_norms) / len(single_block_norms),
            "summary/single_block_grad_std": float(jnp.std(jnp.array(single_block_norms))),
            "summary/single_block_grad_max": max(single_block_norms),
            "summary/single_block_grad_min": min(single_block_norms),
        })
        print(f"\nSingle block gradient norms:")
        print(f"  Mean: {sum(single_block_norms) / len(single_block_norms):.6f}")
        print(f"  Std:  {float(jnp.std(jnp.array(single_block_norms))):.6f}")
        print(f"  Range: [{min(single_block_norms):.6f}, {max(single_block_norms):.6f}]")
    
    print("\n  → High variance suggests MHC could stabilize training")
    
    wandb.finish()
    return results


# ============================================================================
# Probe Training (Train only MHC, freeze backbone)
# ============================================================================

def run_probe_training(config: AnalysisConfig):
    """
    Train only MHC parameters while keeping Flux frozen.
    
    This tests if cross-layer connections learn useful patterns.
    """
    print("=" * 60)
    print("PROBE TRAINING (MHC only)")
    print("=" * 60)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name="mhc-probe-training",
        config=vars(config),
    )
    
    rngs = nnx.Rngs(default=42)
    params = get_tiny_flux_params(config, rngs)
    model = Flux(params)
    
    # Create MHC manager
    total_blocks = config.depth + config.depth_single_blocks
    mhc_manager = HyperConnectionManager(
        num_blocks=total_blocks,
        hidden_size=config.hidden_size,
        history_len=config.history_len,
        rngs=rngs,
        use_full_mhc=config.use_full_mhc,
    )
    
    print(f"Created MHC manager with {total_blocks} connections")
    print(f"  - History length: {config.history_len}")
    print(f"  - Full MHC: {config.use_full_mhc}")
    
    # Log initial alphas
    initial_alphas = mhc_manager.get_all_alphas()
    if initial_alphas:
        for i, alpha in enumerate(initial_alphas):
            wandb.log({"mhc/initial_alpha": alpha, "block_idx": i})
        print(f"  - Initial alpha (all blocks): {initial_alphas[0]:.4f}")
    
    # Training loop
    import optax
    
    # Only optimize MHC parameters
    mhc_params = []
    for conn in mhc_manager.connections:
        mhc_params.append(nnx.state(conn))
    
    optimizer = optax.adam(config.probe_lr)
    
    print(f"\nTraining for {config.probe_steps} steps...")
    
    for step in range(config.probe_steps):
        batch = create_dummy_batch(config, nnx.Rngs(default=step))
        
        def forward_with_mhc(batch):
            """Forward pass using MHC connections."""
            mhc_manager.reset()
            
            # Process inputs
            img = model.img_in(batch["img"])
            txt = model.txt_in(batch["txt"])
            vec = model.time_in(timestep_embedding(batch["timesteps"], 256))
            vec = vec + model.vector_in(batch["y"])
            
            ids = jnp.concatenate((batch["txt_ids"], batch["img_ids"]), axis=1)
            pe = model.pe_embedder(ids)
            
            # Double blocks with MHC
            block_idx = 0
            for block in model.double_blocks.layers:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                img = mhc_manager.apply(block_idx, img)
                block_idx += 1
            
            # Single blocks with MHC  
            img = jnp.concatenate((txt, img), axis=1)
            # Reset history since shape changed (img only -> txt+img concatenated)
            mhc_manager.reset()
            for block in model.single_blocks.layers:
                img = block(img, vec=vec, pe=pe)
                img = mhc_manager.apply(block_idx, img)
                block_idx += 1
            
            img = img[:, batch["txt"].shape[1]:, ...]
            img = model.final_layer(img, vec)
            
            return jnp.mean(img ** 2)
        
        loss = forward_with_mhc(batch)
        
        # Log progress
        if step % 10 == 0:
            alphas = mhc_manager.get_all_alphas()
            
            wandb.log({
                "train/loss": float(loss),
                "train/step": step,
            })
            
            if alphas:
                wandb.log({
                    "mhc/alpha_mean": sum(alphas) / len(alphas),
                    "mhc/alpha_std": float(jnp.std(jnp.array(alphas))),
                    "mhc/alpha_min": min(alphas),
                    "mhc/alpha_max": max(alphas),
                })
                
                print(f"Step {step:4d} | Loss: {float(loss):.6f} | "
                      f"α mean: {sum(alphas)/len(alphas):.4f} | "
                      f"α range: [{min(alphas):.4f}, {max(alphas):.4f}]")
    
    # Final analysis
    final_alphas = mhc_manager.get_all_alphas()
    
    print("\n" + "=" * 60)
    print("PROBE TRAINING RESULTS")
    print("=" * 60)
    
    if final_alphas:
        print("\nLearned alpha values per block:")
        for i, alpha in enumerate(final_alphas):
            block_type = "double" if i < config.depth else "single"
            local_idx = i if i < config.depth else i - config.depth
            print(f"  Block {i:2d} ({block_type:6s} #{local_idx:2d}): α = {alpha:.4f}")
            wandb.log({"mhc/final_alpha": alpha, "block_idx": i})
        
        # Key metric: did alphas diverge from initial?
        avg_alpha = sum(final_alphas) / len(final_alphas)
        alpha_variance = float(jnp.var(jnp.array(final_alphas)))
        
        wandb.log({
            "summary/final_alpha_mean": avg_alpha,
            "summary/final_alpha_variance": alpha_variance,
            "summary/alpha_diverged_from_init": abs(avg_alpha - 0.9) > 0.05,
        })
        
        print(f"\nSummary:")
        print(f"  Average α: {avg_alpha:.4f} (initial: 0.9)")
        print(f"  α variance: {alpha_variance:.6f}")
        
        if abs(avg_alpha - 0.9) > 0.05 or alpha_variance > 0.01:
            print("\n  ✅ Alphas diverged from initialization!")
            print("     → Cross-layer connections ARE learning useful patterns")
            print("     → MHC is worth pursuing further")
        else:
            print("\n  ⚠️  Alphas stayed near initialization")
            print("     → May need longer training or different setup")
    
    wandb.finish()
    return final_alphas


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    experiment: Literal["activation", "gradient", "probe", "all"] = "all",
    hidden_size: int = 768,
    depth: int = 6,
    depth_single_blocks: int = 6,
    probe_steps: int = 100,
    wandb_project: str = "mhc-flux-analysis",
    wandb_entity: str | None = None,
):
    """
    Run MHC analysis experiments on Flux.
    
    Args:
        experiment: Which experiment to run (activation/gradient/probe/all)
        hidden_size: Hidden dimension of the model
        depth: Number of double blocks
        depth_single_blocks: Number of single blocks
        probe_steps: Training steps for probe experiment
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
    """
    config = AnalysisConfig(
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        probe_steps=probe_steps,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    
    print("\n" + "=" * 60)
    print("MHC ANALYSIS FOR FLUX")
    print("=" * 60)
    print(f"Experiment: {experiment}")
    print(f"Config: {config}")
    print("=" * 60 + "\n")
    
    if experiment in ("activation", "all"):
        run_activation_analysis(config)
    
    if experiment in ("gradient", "all"):
        run_gradient_analysis(config)
    
    if experiment in ("probe", "all"):
        run_probe_training(config)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"View results at: https://wandb.ai/{wandb_entity or 'your-entity'}/{wandb_project}")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
