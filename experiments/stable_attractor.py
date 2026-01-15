"""
Stable Attractor Experiment

Demonstrates how the "Old Money" agent maintains long-term stability
by keeping the world model in a stable attractor state.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world_model import GlobalWorldModel
from subworlds.finance import FinanceSubWorld
from subworlds.social import SocialSubWorld
from subworlds.intervention import InterventionSubWorld
from agents.old_money_agent import OldMoneyAgent
from utils.visualization import Visualizer, plot_stability
from utils.metrics import compute_stability_metrics, compute_attractor_metrics


def run_stable_attractor_experiment(
    num_timesteps: int = 500,
    intervention_frequency: int = 10,
    visualize: bool = True,
    save_dir: str = "./results"
):
    """
    Run stable attractor experiment.
    
    This experiment shows:
    1. World model without intervention becomes unstable
    2. With "Old Money" agent intervention, stability is maintained
    3. Long-term attractor properties emerge
    
    Args:
        num_timesteps: Number of simulation timesteps
        intervention_frequency: How often agent intervenes
        visualize: Whether to create visualizations
        save_dir: Directory to save results
    """
    print("=" * 80)
    print("STABLE ATTRACTOR EXPERIMENT")
    print("=" * 80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== Setup World Model ==========
    print("Setting up world model...")
    
    # Global world model
    world = GlobalWorldModel(
        state_dim=256,
        latent_dim=64,
        hidden_dim=256,
        num_diffusion_steps=1000,
        device=device
    )
    
    # Create sub-worlds
    finance = FinanceSubWorld(
        global_world_model=world,
        name="finance",
        num_assets=10,
        latent_dim=32,
        device=device
    )
    
    social = SocialSubWorld(
        global_world_model=world,
        name="social",
        num_entities=20,
        latent_dim=32,
        device=device
    )
    
    intervention = InterventionSubWorld(
        global_world_model=world,
        name="intervention",
        num_resources=5,
        num_capacities=5,
        latent_dim=32,
        device=device
    )
    
    # Register sub-worlds
    world.register_subworld(finance)
    world.register_subworld(social)
    world.register_subworld(intervention)
    
    print(f"World model: {world}")
    print()
    
    # ========== Setup Agent ==========
    print("Setting up Old Money agent...")
    
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
    
    print(f"Agent: {agent}")
    print()
    
    # ========== Run Simulation ==========
    print("Running simulation...")
    print(f"Timesteps: {num_timesteps}")
    print(f"Intervention frequency: every {intervention_frequency} steps")
    print()
    
    # Initialize
    world.initialize(batch_size=1)
    
    # Track metrics
    stability_scores = []
    intervention_costs = []
    
    for t in range(num_timesteps):
        # Get world state
        world_state = world.get_all_observable_states()
        
        # Agent intervention (periodic)
        if t % intervention_frequency == 0:
            action = agent.intervene(world_state, enforce_budget=True)
            if action is not None:
                intervention_costs.append(torch.norm(action).item())
        else:
            action = None
        
        # Step world
        interventions = {intervention.name: action} if action is not None else {}
        world.step(interventions)
        
        # Compute stability
        stability = world.compute_global_stability(window=min(50, t+1))
        stability_scores.append(stability)
        
        # Progress
        if (t + 1) % 100 == 0:
            print(f"Step {t+1}/{num_timesteps} | Stability: {stability:.4f} | "
                  f"Avg Cost: {np.mean(intervention_costs[-10:]) if intervention_costs else 0:.4f}")
    
    print()
    print("Simulation complete!")
    print()
    
    # ========== Analyze Results ==========
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Overall stability
    avg_stability = np.mean(stability_scores)
    final_stability = np.mean(stability_scores[-50:])
    initial_stability = np.mean(stability_scores[:50])
    
    print(f"Average Stability: {avg_stability:.4f}")
    print(f"Initial Stability (first 50 steps): {initial_stability:.4f}")
    print(f"Final Stability (last 50 steps): {final_stability:.4f}")
    print(f"Stability Improvement: {final_stability - initial_stability:.4f}")
    print()
    
    # Intervention efficiency
    total_cost = sum(intervention_costs)
    avg_cost = np.mean(intervention_costs) if intervention_costs else 0
    efficiency = final_stability / (avg_cost + 1e-6)
    
    print(f"Total Intervention Cost: {total_cost:.4f}")
    print(f"Average Intervention Cost: {avg_cost:.4f}")
    print(f"Intervention Efficiency: {efficiency:.4f}")
    print()
    
    # Attractor metrics
    print("Computing attractor metrics...")
    finance_metrics = compute_attractor_metrics(finance.get_history())
    social_metrics = compute_attractor_metrics(social.get_history())
    
    print(f"Finance Sub-world:")
    print(f"  Attractor Strength: {finance_metrics.get('attractor_strength', 0):.4f}")
    print(f"  Basin Size: {finance_metrics.get('basin_size', 0):.4f}")
    print(f"  Intrinsic Dimensionality: {finance_metrics.get('intrinsic_dimensionality', 0)}")
    print()
    
    print(f"Social Sub-world:")
    print(f"  Attractor Strength: {social_metrics.get('attractor_strength', 0):.4f}")
    print(f"  Basin Size: {social_metrics.get('basin_size', 0):.4f}")
    print(f"  Intrinsic Dimensionality: {social_metrics.get('intrinsic_dimensionality', 0)}")
    print()
    
    # ========== Visualizations ==========
    if visualize:
        print("Creating visualizations...")
        vis = Visualizer()
        
        # Plot stability over time
        plot_stability(
            stability_scores,
            title="Global Stability Over Time (with Old Money Agent)"
        )
        
        # Plot sub-world evolution
        vis.plot_subworld_evolution(
            finance,
            components=list(range(5)),
            save_path=os.path.join(save_dir, "finance_evolution.png")
        )
        
        vis.plot_subworld_evolution(
            social,
            components=list(range(5)),
            save_path=os.path.join(save_dir, "social_evolution.png")
        )
        
        # Plot intervention impact
        vis.plot_intervention_impact(
            agent,
            intervention,
            save_path=os.path.join(save_dir, "intervention_impact.png")
        )
        
        # Plot phase space
        vis.plot_phase_space(
            finance,
            dim1=0,
            dim2=1,
            save_path=os.path.join(save_dir, "finance_phase_space.png")
        )
        
        print(f"Visualizations saved to {save_dir}")
        print()
    
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return {
        'world': world,
        'agent': agent,
        'stability_scores': stability_scores,
        'intervention_costs': intervention_costs,
        'metrics': {
            'avg_stability': avg_stability,
            'final_stability': final_stability,
            'stability_improvement': final_stability - initial_stability,
            'efficiency': efficiency,
            'finance_attractor': finance_metrics,
            'social_attractor': social_metrics,
        }
    }


if __name__ == "__main__":
    results = run_stable_attractor_experiment(
        num_timesteps=500,
        intervention_frequency=10,
        visualize=True,
        save_dir="./results/stable_attractor"
    )

