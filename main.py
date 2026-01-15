"""
Main entry point for the Old Money world model.

This script provides a simple interface to run experiments and simulations.
"""

import torch
import argparse
import yaml
import os
import sys

from core.world_model import GlobalWorldModel
from subworlds.finance import FinanceSubWorld
from subworlds.social import SocialSubWorld
from subworlds.intervention import InterventionSubWorld
from agents.old_money_agent import OldMoneyAgent
from experiments import (
    run_stable_attractor_experiment,
    run_intervention_demo,
    run_multi_agent_experiment
)


def load_config(config_path: str = "configs/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_world_from_config(config: dict):
    """Create world model from configuration."""
    device = config['world_model']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Create world model
    world = GlobalWorldModel(
        state_dim=config['world_model']['state_dim'],
        latent_dim=config['world_model']['latent_dim'],
        hidden_dim=config['world_model']['hidden_dim'],
        num_diffusion_steps=config['world_model']['num_diffusion_steps'],
        device=device
    )
    
    # Create sub-worlds
    finance = FinanceSubWorld(
        global_world_model=world,
        name="finance",
        num_assets=config['subworlds']['finance']['num_assets'],
        latent_dim=config['subworlds']['finance']['latent_dim'],
        device=device
    )
    
    social = SocialSubWorld(
        global_world_model=world,
        name="social",
        num_entities=config['subworlds']['social']['num_entities'],
        latent_dim=config['subworlds']['social']['latent_dim'],
        device=device
    )
    
    intervention = InterventionSubWorld(
        global_world_model=world,
        name="intervention",
        num_resources=config['subworlds']['intervention']['num_resources'],
        num_capacities=config['subworlds']['intervention']['num_capacities'],
        latent_dim=config['subworlds']['intervention']['latent_dim'],
        device=device
    )
    
    # Register sub-worlds
    world.register_subworld(finance)
    world.register_subworld(social)
    world.register_subworld(intervention)
    
    # Create agent
    agent = OldMoneyAgent(
        name=config['agent']['name'],
        intervention_subworld=intervention,
        observed_subworlds=[finance, social],
        action_dim=config['agent']['action_dim'],
        hidden_dim=config['agent']['hidden_dim'],
        num_diffusion_steps=config['agent']['num_diffusion_steps'],
        horizon=config['agent']['horizon'],
        prior_type=config['agent']['prior_type'],
        device=device
    )
    
    return world, agent, (finance, social, intervention)


def main():
    parser = argparse.ArgumentParser(
        description="Old Money: Diffusion-Based Hierarchical World Model"
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='stable_attractor',
        choices=['stable_attractor', 'intervention_demo', 'multi_agent', 'custom'],
        help='Experiment to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500,
        help='Number of simulation timesteps'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualizations'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = None
    
    # Run experiment
    print("=" * 80)
    print("OLD MONEY: Diffusion-Based Hierarchical World Model")
    print("=" * 80)
    print()
    
    if args.experiment == 'stable_attractor':
        results = run_stable_attractor_experiment(
            num_timesteps=args.timesteps,
            intervention_frequency=10,
            visualize=not args.no_viz,
            save_dir=os.path.join(args.save_dir, 'stable_attractor')
        )
    
    elif args.experiment == 'intervention_demo':
        results = run_intervention_demo(
            num_timesteps=args.timesteps,
            shock_timestep=args.timesteps // 3,
            shock_magnitude=2.0,
            visualize=not args.no_viz,
            save_dir=os.path.join(args.save_dir, 'intervention_demo')
        )
    
    elif args.experiment == 'multi_agent':
        results = run_multi_agent_experiment(
            num_timesteps=args.timesteps,
            num_agents=2,
            visualize=not args.no_viz,
            save_dir=os.path.join(args.save_dir, 'multi_agent')
        )
    
    elif args.experiment == 'custom':
        if config is None:
            print("Error: Config file required for custom experiment")
            return
        
        world, agent, subworlds = create_world_from_config(config)
        print("Custom world model created successfully!")
        print(f"World: {world}")
        print(f"Agent: {agent}")
        print()
        print("You can now use the world model and agent for custom experiments.")
        
        # Example: run a simple simulation
        print("Running simple simulation...")
        world.initialize(batch_size=1)
        
        for t in range(100):
            world_state = world.get_all_observable_states()
            action = agent.intervene(world_state)
            interventions = {subworlds[2].name: action} if action is not None else {}
            world.step(interventions)
            
            if (t + 1) % 20 == 0:
                stability = world.compute_global_stability()
                print(f"Step {t+1}: Stability = {stability:.4f}")
        
        print("Simulation complete!")
    
    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

