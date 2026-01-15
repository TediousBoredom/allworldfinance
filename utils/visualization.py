"""
Visualization tools for world model dynamics and agent behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """
    Comprehensive visualization toolkit for the world model.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_subworld_evolution(
        self,
        subworld,
        components: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot evolution of sub-world state components over time.
        
        Args:
            subworld: SubWorld instance
            components: List of component indices to plot (None = all)
            figsize: Figure size
            save_path: Path to save figure
        """
        history = subworld.get_history()
        if not history:
            print("No history available")
            return
        
        # Extract states
        states = torch.stack([s.state for s in history]).squeeze()
        timesteps = np.arange(len(states))
        
        # Select components
        if components is None:
            components = list(range(min(10, states.shape[-1])))
        
        fig, axes = plt.subplots(len(components), 1, figsize=figsize, sharex=True)
        if len(components) == 1:
            axes = [axes]
        
        for idx, comp in enumerate(components):
            ax = axes[idx]
            values = states[:, comp].cpu().numpy()
            ax.plot(timesteps, values, linewidth=2, color=self.colors[idx % len(self.colors)])
            ax.set_ylabel(f'Component {comp}', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestep', fontsize=12)
        fig.suptitle(f'{subworld.name.capitalize()} Sub-world Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stability_over_time(
        self,
        world_model,
        window: int = 50,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot stability metrics over time for all sub-worlds.
        
        Args:
            world_model: GlobalWorldModel instance
            window: Window size for stability computation
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, subworld in world_model.subworlds.items():
            history = subworld.get_history()
            if len(history) < window:
                continue
            
            # Compute stability at each timestep
            stabilities = []
            for i in range(window, len(history)):
                # Compute stability for window ending at i
                recent = history[i-window:i]
                changes = []
                for j in range(1, len(recent)):
                    change = torch.norm(recent[j].state - recent[j-1].state).item()
                    changes.append(change)
                variance = np.var(changes)
                stability = 1.0 / (1.0 + variance)
                stabilities.append(stability)
            
            timesteps = np.arange(window, len(history))
            ax.plot(timesteps, stabilities, label=name, linewidth=2)
        
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Stability Score', fontsize=12)
        ax.set_title('Sub-world Stability Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_intervention_impact(
        self,
        agent,
        intervention_subworld,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot impact of agent interventions.
        
        Args:
            agent: OldMoneyAgent instance
            intervention_subworld: InterventionSubWorld instance
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Action magnitudes over time
        actions = agent.get_action_history()
        if actions:
            action_norms = [torch.norm(a).item() for a in actions]
            axes[0].plot(action_norms, linewidth=2, color=self.colors[0])
            axes[0].set_ylabel('Action Magnitude', fontsize=11)
            axes[0].set_title('Intervention Strength', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Resource levels over time
        history = intervention_subworld.get_history()
        if history:
            resources = torch.stack([s.state[:, :intervention_subworld.num_resources] 
                                    for s in history]).squeeze()
            for i in range(min(5, resources.shape[-1])):
                axes[1].plot(resources[:, i].cpu().numpy(), 
                           label=f'Resource {i}', linewidth=2)
            axes[1].set_ylabel('Resource Level', fontsize=11)
            axes[1].set_title('Resource Dynamics', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=9, ncol=5)
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Stability score over time
        if agent.stability_history:
            axes[2].plot(agent.stability_history, linewidth=2, color=self.colors[2])
            axes[2].axhline(y=np.mean(agent.stability_history), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(agent.stability_history):.3f}')
            axes[2].set_ylabel('Stability Score', fontsize=11)
            axes[2].set_xlabel('Timestep', fontsize=11)
            axes[2].set_title('System Stability', fontsize=12, fontweight='bold')
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_space(
        self,
        subworld,
        dim1: int = 0,
        dim2: int = 1,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot phase space trajectory of sub-world.
        
        Args:
            subworld: SubWorld instance
            dim1: First dimension to plot
            dim2: Second dimension to plot
            figsize: Figure size
            save_path: Path to save figure
        """
        history = subworld.get_history()
        if not history:
            print("No history available")
            return
        
        states = torch.stack([s.state for s in history]).squeeze()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectory
        x = states[:, dim1].cpu().numpy()
        y = states[:, dim2].cpu().numpy()
        
        # Color by time
        colors = np.arange(len(x))
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=20, alpha=0.6)
        
        # Plot trajectory line
        ax.plot(x, y, 'k-', alpha=0.2, linewidth=1)
        
        # Mark start and end
        ax.scatter(x[0], y[0], c='green', s=200, marker='o', 
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], c='red', s=200, marker='s', 
                  edgecolors='black', linewidths=2, label='End', zorder=5)
        
        ax.set_xlabel(f'Dimension {dim1}', fontsize=12)
        ax.set_ylabel(f'Dimension {dim2}', fontsize=12)
        ax.set_title(f'{subworld.name.capitalize()} Phase Space', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Timestep', fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_dashboard(
        self,
        world_model,
        agent,
        save_path: Optional[str] = None
    ):
        """
        Create interactive dashboard using Plotly.
        
        Args:
            world_model: GlobalWorldModel instance
            agent: OldMoneyAgent instance
            save_path: Path to save HTML
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Global Stability',
                'Intervention Actions',
                'Finance Sub-world',
                'Social Sub-world',
                'Resource Levels',
                'Agent Stability'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Plot 1: Global stability
        timesteps = list(range(len(world_model.global_state_history)))
        if timesteps:
            fig.add_trace(
                go.Scatter(x=timesteps, y=[world_model.compute_global_stability(50) 
                          for _ in timesteps],
                          mode='lines', name='Global Stability',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Plot 2: Intervention actions
        actions = agent.get_action_history()
        if actions:
            action_norms = [torch.norm(a).item() for a in actions]
            fig.add_trace(
                go.Scatter(x=list(range(len(action_norms))), y=action_norms,
                          mode='lines', name='Action Magnitude',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )
        
        # Add more plots for other sub-worlds...
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Old Money World Model Dashboard",
            title_font_size=20
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()


def plot_trajectories(
    states: torch.Tensor,
    labels: Optional[List[str]] = None,
    title: str = "State Trajectories",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Quick plot of state trajectories.
    
    Args:
        states: State tensor [timesteps, num_components]
        labels: Component labels
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    states_np = states.cpu().numpy()
    timesteps = np.arange(len(states_np))
    
    for i in range(states_np.shape[1]):
        label = labels[i] if labels and i < len(labels) else f'Component {i}'
        ax.plot(timesteps, states_np[:, i], label=label, linewidth=2)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stability(
    stability_scores: List[float],
    title: str = "Stability Over Time",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Quick plot of stability scores.
    
    Args:
        stability_scores: List of stability scores
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    timesteps = np.arange(len(stability_scores))
    ax.plot(timesteps, stability_scores, linewidth=2, color='blue')
    ax.axhline(y=np.mean(stability_scores), color='red', 
              linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(stability_scores):.3f}')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Stability Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

