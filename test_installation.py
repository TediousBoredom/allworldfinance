#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality.
"""

import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.world_model import GlobalWorldModel
        from core.diffusion import DiffusionModel, DDPMScheduler
        from subworlds.finance import FinanceSubWorld
        from subworlds.social import SocialSubWorld
        from subworlds.intervention import InterventionSubWorld
        from agents.old_money_agent import OldMoneyAgent
        from agents.priors import AdaptivePrior, StabilityPrior
        from utils.visualization import Visualizer
        from utils.metrics import compute_stability_metrics
        from utils.causal_graph import CausalGraph
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create world model
        world = GlobalWorldModel(
            state_dim=128,
            latent_dim=32,
            hidden_dim=128,
            device=device
        )
        print("✓ World model created")
        
        # Create sub-worlds
        finance = FinanceSubWorld(world, name="finance", num_assets=5, device=device)
        social = SocialSubWorld(world, name="social", num_entities=10, device=device)
        intervention = InterventionSubWorld(world, name="intervention", device=device)
        print("✓ Sub-worlds created")
        
        # Register sub-worlds
        world.register_subworld(finance)
        world.register_subworld(social)
        world.register_subworld(intervention)
        print("✓ Sub-worlds registered")
        
        # Create agent
        agent = OldMoneyAgent(
            intervention_subworld=intervention,
            observed_subworlds=[finance, social],
            action_dim=intervention.state_dim,
            device=device
        )
        print("✓ Agent created")
        
        # Initialize and run a few steps
        world.initialize(batch_size=1)
        print("✓ World initialized")
        
        for t in range(5):
            world_state = world.get_all_observable_states()
            action = agent.intervene(world_state)
            interventions = {intervention.name: action} if action is not None else {}
            world.step(interventions)
        
        print("✓ Simulation steps completed")
        
        # Check stability
        stability = world.compute_global_stability()
        print(f"✓ Stability computed: {stability:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        import os
        
        config_path = "configs/default.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration loaded")
            print(f"  - World state dim: {config['world_model']['state_dim']}")
            print(f"  - Agent prior type: {config['agent']['prior_type']}")
            return True
        else:
            print("⚠ Configuration file not found (optional)")
            return True
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OLD MONEY - Installation Test")
    print("=" * 60)
    print()
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Configuration", test_configuration()))
    
    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Installation successful.")
        print()
        print("Next steps:")
        print("  1. Run quickstart: python quickstart.py")
        print("  2. Run experiments: ./run.sh stable")
        print("  3. Explore the code in core/, subworlds/, and agents/")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

