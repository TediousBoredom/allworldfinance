"""
Intervention Demonstration

Shows how the Old Money agent adapts its intervention strategy
based on observed states of other sub-worlds.
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
from utils.metrics import compute_causal_impact
import matplotlib.pyplot as plt


def run_intervention_demo(
    num_timesteps: int = 300,
    shock_timestep: int = 100,
    shock_magnitude: float = 2.0,
    visualize: bool = True,
    save_dir: str = "./results"
):
    """
    Demonstrate adaptive intervention in response to external shocks.
    
    This experiment:
    1. Runs world model in stable state
    2. Introduces external shock to finance sub-world
    3. Shows how Old Money agent adapts intervention to restore stability
    
    Args:
        num_timesteps: Number of simulation timesteps
        shock_timestep: When to introduce shock
        shock_magnitude: Magnitude of shock
        visualize: Whether to create visualizations
        save_dir: Directory to save results
    """
    print("=" * 80)
    print("ADAPTIVE INTERVENTION DEMONSTRATION")
    print("=" * 80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== Setup ==========
    print("Setting up world model and agent...")
    
    world = GlobalWorldModel(
        state_dim=256,
        latent_dim=64,
        hidden_dim=256,
        device=device
    )
    
    finance = FinanceSubWorld(world, name="finance", num_assets=10, device=device)
    social = SocialSubWorld(world, name="social", num_entities=20, device=device)
    intervention = InterventionSubWorld(world, name="intervention", device=device)
    
    world.register_subworld(finance)
    world.register_subworld(social)
    world.register_subworld(intervention)
    
    agent = OldMoneyAgent(
        name="old_money",
        intervention_subworld=intervention,
        observed_subworlds=[finance, social],
        action_dim=intervention.state_dim,
        prior_type="adaptive",
        device=device
    )
    
    world.initialize(batch_size=1)
    
    print("Setup complete!")
    print()
    
    # ========== Run Simulation ==========
    print(f"Running simulation with shock at t={shock_timestep}...")
    print()
    
    stability_scores = []
    finance_volatility = []
    social_trust = []
    intervention_magnitudes = []
    
    for t in range(num_timesteps):
        # Introduce shock
        if t == shock_timestep:
            print(f"⚡ SHOCK at timestep {t}! Introducing market volatility...")
            # Inject shock into finance sub-world
            finance_state = finance.get_state()
            shock = torch.randn_like(finance_state.state) * shock_magnitude
            finance_state.state = finance_state.state + shock
            finance.set_state(finance_state)
        
        # Agent observes and intervenes
        world_state = world.get_all_observable_states()
        action = agent.intervene(world_state, enforce_budget=True)
        
        if action is not None:
            intervention_magnitudes.append(torch.norm(action).item())
        else:
            intervention_magnitudes.append(0.0)
        
        # Step world
        interventions = {intervention.name: action} if action is not None else {}
        world.step(interventions)
        
        # Track metrics
        stability = world.compute_global_stability(window=min(50, t+1))
        stability_scores.append(stability)
        
        finance_volatility.append(finance.get_volatility().mean().item())
        social_trust.append(social.get_trust_matrix().mean().item())
        
        if (t + 1) % 50 == 0:
            print(f"Step {t+1}/{num_timesteps} | Stability: {stability:.4f} | "
                  f"Volatility: {finance_volatility[-1]:.4f}")
    
    print()
    print("Simulation complete!")
    print()
    
    # ========== Analysis ==========
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    # Pre-shock vs post-shock stability
    pre_shock = stability_scores[:shock_timestep]
    post_shock = stability_scores[shock_timestep:shock_timestep+50]
    recovery = stability_scores[shock_timestep+50:]
    
    print(f"Pre-shock stability: {np.mean(pre_shock):.4f}")
    print(f"Immediate post-shock stability: {np.mean(post_shock):.4f}")
    if recovery:
        print(f"Recovery stability: {np.mean(recovery):.4f}")
    print()
    
    # Intervention response
    pre_shock_intervention = intervention_magnitudes[:shock_timestep]
    post_shock_intervention = intervention_magnitudes[shock_timestep:shock_timestep+50]
    
    print(f"Pre-shock avg intervention: {np.mean(pre_shock_intervention):.4f}")
    print(f"Post-shock avg intervention: {np.mean(post_shock_intervention):.4f}")
    print(f"Intervention increase: {np.mean(post_shock_intervention) - np.mean(pre_shock_intervention):.4f}")
    print()
    
    # Recovery time
    recovery_threshold = np.mean(pre_shock) * 0.95
    recovery_idx = None
    for i in range(shock_timestep, len(stability_scores)):
        if stability_scores[i] >= recovery_threshold:
            recovery_idx = i
            break
    
    if recovery_idx:
        recovery_time = recovery_idx - shock_timestep
        print(f"Recovery time: {recovery_time} timesteps")
    else:
        print("System did not fully recover within simulation time")
    print()
    
    # ========== Visualizations ==========
    if visualize:
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot 1: Stability
        axes[0].plot(stability_scores, linewidth=2, color='blue')
        axes[0].axvline(x=shock_timestep, color='red', linestyle='--', 
                       linewidth=2, label='Shock')
        axes[0].set_ylabel('Stability', fontsize=11)
        axes[0].set_title('System Stability', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Finance volatility
        axes[1].plot(finance_volatility, linewidth=2, color='orange')
        axes[1].axvline(x=shock_timestep, color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('Volatility', fontsize=11)
        axes[1].set_title('Market Volatility', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Social trust
        axes[2].plot(social_trust, linewidth=2, color='green')
        axes[2].axvline(x=shock_timestep, color='red', linestyle='--', linewidth=2)
        axes[2].set_ylabel('Trust Level', fontsize=11)
        axes[2].set_title('Social Trust', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Intervention magnitude
        axes[3].plot(intervention_magnitudes, linewidth=2, color='purple')
        axes[3].axvline(x=shock_timestep, color='red', linestyle='--', linewidth=2)
        axes[3].set_ylabel('Intervention', fontsize=11)
        axes[3].set_xlabel('Timestep', fontsize=11)
        axes[3].set_title('Agent Intervention Magnitude', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "intervention_response.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {save_dir}")
        print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    return {
        'world': world,
        'agent': agent,
        'stability_scores': stability_scores,
        'recovery_time': recovery_time if recovery_idx else None,
    }


if __name__ == "__main__":
    results = run_intervention_demo(
        num_timesteps=300,
        shock_timestep=100,
        shock_magnitude=2.0,
        visualize=True,
        save_dir="./results/intervention_demo"
    )

