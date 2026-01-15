"""
Quick start example for the Old Money world model.

This script demonstrates the basic usage of the framework.
"""

import torch
from core.world_model import GlobalWorldModel
from subworlds.finance import FinanceSubWorld
from subworlds.social import SocialSubWorld
from subworlds.intervention import InterventionSubWorld
from agents.old_money_agent import OldMoneyAgent
from utils.visualization import Visualizer, plot_stability


def main():
    print("=" * 80)
    print("OLD MONEY QUICK START")
    print("=" * 80)
    print()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Step 1: Create global world model
    print("Step 1: Creating global world model...")
    world = GlobalWorldModel(
        state_dim=256,
        latent_dim=64,
        hidden_dim=256,
        num_diffusion_steps=1000,
        device=device
    )
    print(f"✓ World model created: {world}")
    print()
    
    # Step 2: Create sub-worlds
    print("Step 2: Creating sub-worlds...")
    
    finance = FinanceSubWorld(
        global_world_model=world,
        name="finance",
        num_assets=10,
        latent_dim=32,
        device=device
    )
    print(f"✓ Finance sub-world created")
    
    social = SocialSubWorld(
        global_world_model=world,
        name="social",
        num_entities=20,
        latent_dim=32,
        device=device
    )
    print(f"✓ Social sub-world created")
    
    intervention = InterventionSubWorld(
        global_world_model=world,
        name="intervention",
        num_resources=5,
        num_capacities=5,
        latent_dim=32,
        device=device
    )
    print(f"✓ Intervention sub-world created")
    print()
    
    # Step 3: Register sub-worlds
    print("Step 3: Registering sub-worlds with global model...")
    world.register_subworld(finance)
    world.register_subworld(social)
    world.register_subworld(intervention)
    print("✓ All sub-worlds registered")
    print()
    
    # Step 4: Create Old Money agent
    print("Step 4: Creating Old Money agent...")
    agent = OldMoneyAgent(
        name="old_money",
        intervention_subworld=intervention,
        observed_subworlds=[finance, social],
        action_dim=intervention.state_dim,
        hidden_dim=256,
        num_diffusion_steps=50,
        horizon=20,
        prior_type="stability",
        device=device
    )
    print(f"✓ Agent created: {agent}")
    print()
    
    # Step 5: Initialize and run simulation
    print("Step 5: Running simulation...")
    world.initialize(batch_size=1)
    
    num_timesteps = 200
    stability_scores = []
    
    for t in range(num_timesteps):
        # Agent observes and intervenes
        world_state = world.get_all_observable_states()
        action = agent.intervene(world_state, enforce_budget=True)
        
        # Step world
        interventions = {intervention.name: action} if action is not None else {}
        world.step(interventions)
        
        # Track stability
        stability = world.compute_global_stability(window=min(50, t+1))
        stability_scores.append(stability)
        
        # Progress
        if (t + 1) % 50 == 0:
            print(f"  Step {t+1}/{num_timesteps} | Stability: {stability:.4f}")
    
    print("✓ Simulation complete!")
    print()
    
    # Step 6: Analyze results
    print("Step 6: Analyzing results...")
    import numpy as np
    
    avg_stability = np.mean(stability_scores)
    final_stability = np.mean(stability_scores[-50:])
    
    print(f"  Average stability: {avg_stability:.4f}")
    print(f"  Final stability: {final_stability:.4f}")
    print(f"  Agent long-term stability: {agent.get_long_term_stability():.4f}")
    print()
    
    # Step 7: Visualize
    print("Step 7: Creating visualization...")
    plot_stability(
        stability_scores,
        title="System Stability with Old Money Agent"
    )
    print("✓ Visualization displayed")
    print()
    
    print("=" * 80)
    print("QUICK START COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run full experiments: python main.py --experiment stable_attractor")
    print("  2. Try different configurations: python main.py --config configs/custom.yaml")
    print("  3. Explore the codebase in core/, subworlds/, and agents/")
    print()


if __name__ == "__main__":
    main()

