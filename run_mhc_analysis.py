"""
Quick start script for MHC experiments.

Sets up W&B and runs all analyses.
"""

import os
import subprocess
import sys


def setup_wandb():
    """Configure W&B with API key."""
    # Set API key from environment or prompt
    api_key = os.environ.get("WANDB_API_KEY")
    
    if not api_key:
        print("=" * 60)
        print("W&B SETUP")
        print("=" * 60)
        print("No WANDB_API_KEY found in environment.")
        print("Please set it using:")
        print('  $env:WANDB_API_KEY = "your_key_here"')
        print("\nOr run: wandb login")
        print("=" * 60)
        
        # Try wandb login
        try:
            subprocess.run(["wandb", "login"], check=True)
        except Exception as e:
            print(f"Could not run wandb login: {e}")
            return False
    
    return True


def run_experiments():
    """Run all MHC analysis experiments."""
    from jflux.analysis import main
    
    # Run all experiments with tiny model for fast iteration
    main(
        experiment="all",
        hidden_size=768,
        depth=6,
        depth_single_blocks=6,
        probe_steps=100,
        wandb_project="mhc-flux-analysis",
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MHC-FLUX ANALYSIS RUNNER")
    print("=" * 60)
    
    # Check if wandb is available
    try:
        import wandb
        print("✓ wandb available")
    except ImportError:
        print("Installing wandb...")
        subprocess.run([sys.executable, "-m", "pip", "install", "wandb"])
    
    # Check if optax is available
    try:
        import optax
        print("✓ optax available")
    except ImportError:
        print("Installing optax...")
        subprocess.run([sys.executable, "-m", "pip", "install", "optax"])
    
    # Setup wandb
    if setup_wandb():
        run_experiments()
    else:
        print("Please configure W&B and try again.")
