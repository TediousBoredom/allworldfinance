"""
Multi-Agent Experiment

Demonstrates multiple agents with different strategies interacting
in the same world model.
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
from utils.visualization import Visualizer
import matplotlib.pyplot as plt


def run_multi_agent_experiment(
    num_timesteps: int = 400,
    num_agents: int = 2,
    visualize: bool = True,
    save_dir: str = "./results"
):
    """
    Run multi-agent experiment with competing intervention strategies.
    
    This experiment shows:
    1. Multiple agents with different intervention strategies
    2. How they compete or cooperate
    3. Emergent stability properties
    
    Args:
        num_timesteps: Number of simulation timesteps
        num_agents: Number of agents (2 or 3)
        visualize: Whether to create visualizations
        save_dir: Directory to save results
    """
    print("=" * 80)
    print("MULTI-AGENT EXPERIMENT")
    print("=" * 80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Number of agents: {num_agents}")
    print()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== Setup World Model ==========
    print("Setting up world model...")
    
    world = GlobalWorldModel(
        state_dim=512,  # Larger state space for multiple agents
        latent_dim=64,
        hidden_dim=256,
        device=device
    )
    
    # Shared observation sub-worlds
    finance = FinanceSubWorld(world, name="finance", num_assets=10, device=device)
    social = SocialSubWorld(world, name="social", num_entities=20, device=device)
    
    world.register_subworld(finance)
    world.register_subworld(social)
    
    # Create intervention sub-worlds for each agent
    intervention_subworlds = []
    agents = []
    
    for i in range(num_agents):
        intervention = InterventionSubWorld(
            world,
            name=f"intervention_{i}",
            num_resources=5,
            num_capacities=5,
            device=device
        )
        world.register_subworld(intervention)
        intervention_subworlds.append(intervention)
        
        # Create agent with different strategy
        prior_type = "stability" if i == 0 else "adaptive"
        agent = OldMoneyAgent(
            name=f"agent_{i}",
            intervention_subworld=intervention,
            observed_subworlds=[finance, social],
            action_dim=intervention.state_dim,
            prior_type=prior_type,
            horizon=20 + i * 10,  # Different horizons
            device=device
        )
        agents.append(agent)
        
        print(f"Agent {i}: {prior_type} strategy, horizon={20 + i * 10}")
    
    print()
    print(f"World model: {world}")
    print()
    
    # ========== Run Simulation ==========
    print("Running multi-agent simulation...")
    print()
    
    world.initialize(batch_size=1)
    
    # Track metrics per agent
    agent_metrics = {
        i: {
            'stability': [],
            'interventions': [],
            'costs': []
        }
        for i in range(num_agents)
    }
    
    global_stability = []
    
    for t in range(num_timesteps):
        world_state = world.get_all_observable_states()
        
        # Each agent decides on intervention
        interventions = {}
        for i, agent in enumerate(agents):
            action = agent.intervene(world_state, enforce_budget=True)
            
            if action is not None:
                interventions[intervention_subworlds[i].name] = action
                agent_metrics[i]['interventions'].append(torch.norm(action).item())
            else:
                agent_metrics[i]['interventions'].append(0.0)
        
        # Step world with all interventions
        world.step(interventions)
        
        # Track metrics
        stability = world.compute_global_stability(window=min(50, t+1))
        global_stability.append(stability)
        
        for i, agent in enumerate(agents):
            agent_stability = agent.compute_stability_score(window=min(50, t+1))
            agent_metrics[i]['stability'].append(agent_stability)
        
        if (t + 1) % 100 == 0:
            print(f"Step {t+1}/{num_timesteps} | Global Stability: {stability:.4f}")
            for i in range(num_agents):
                avg_intervention = np.mean(agent_metrics[i]['interventions'][-10:])
                print(f"  Agent {i}: Intervention={avg_intervention:.4f}")
    
    print()
    print("Simulation complete!")
    print()
    
    # ========== Analysis ==========
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"Global Stability: {np.mean(global_stability):.4f}")
    print()
    
    for i in range(num_agents):
        avg_stability = np.mean(agent_metrics[i]['stability'])
        avg_intervention = np.mean(agent_metrics[i]['interventions'])
        total_cost = sum(agent_metrics[i]['interventions'])
        efficiency = avg_stability / (avg_intervention + 1e-6)
        
        print(f"Agent {i} ({agents[i].structured_prior.__class__.__name__}):")
        print(f"  Average Stability: {avg_stability:.4f}")
        print(f"  Average Intervention: {avg_intervention:.4f}")
        print(f"  Total Cost: {total_cost:.4f}")
        print(f"  Efficiency: {efficiency:.4f}")
        print()
    
    # Competition vs cooperation analysis
    intervention_correlation = np.corrcoef(
        agent_metrics[0]['interventions'],
        agent_metrics[1]['interventions']
    )[0, 1]
    
    print(f"Intervention Correlation: {intervention_correlation:.4f}")
    if intervention_correlation > 0.5:
        print("  → Agents are cooperating (positive correlation)")
    elif intervention_correlation < -0.5:
        print("  → Agents are competing (negative correlation)")
    else:
        print("  → Agents are independent")
    print()
    
    # ========== Visualizations ==========
    if visualize:
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Global stability
        axes[0].plot(global_stability, linewidth=2, color='blue', label='Global')
        axes[0].set_ylabel('Stability', fontsize=11)
        axes[0].set_title('Global System Stability', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Agent interventions
        colors = ['red', 'green', 'purple']
        for i in range(num_agents):
            axes[1].plot(agent_metrics[i]['interventions'], 
                        linewidth=2, color=colors[i], 
                        label=f'Agent {i}', alpha=0.7)
        axes[1].set_ylabel('Intervention', fontsize=11)
        axes[1].set_title('Agent Intervention Magnitudes', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Agent stability scores
        for i in range(num_agents):
            axes[2].plot(agent_metrics[i]['stability'], 
                        linewidth=2, color=colors[i], 
                        label=f'Agent {i}', alpha=0.7)
        axes[2].set_ylabel('Stability', fontsize=11)
        axes[2].set_xlabel('Timestep', fontsize=11)
        axes[2].set_title('Agent-Specific Stability', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "multi_agent_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Scatter plot: efficiency comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(num_agents):
            avg_cost = np.mean(agent_metrics[i]['interventions'])
            avg_stability = np.mean(agent_metrics[i]['stability'])
            ax.scatter(avg_cost, avg_stability, s=300, 
                      color=colors[i], label=f'Agent {i}',
                      edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Average Intervention Cost', fontsize=12)
        ax.set_ylabel('Average Stability', fontsize=12)
        ax.set_title('Agent Efficiency Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "agent_efficiency.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {save_dir}")
        print()
    
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return {
        'world': world,
        'agents': agents,
        'global_stability': global_stability,
        'agent_metrics': agent_metrics,
    }


if __name__ == "__main__":
    results = run_multi_agent_experiment(
        num_timesteps=400,
        num_agents=2,
        visualize=True,
        save_dir="./results/multi_agent"
    )

