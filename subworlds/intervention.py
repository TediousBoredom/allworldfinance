"""
Intervention sub-world - the "Old Money" agent's domain.

This sub-world is where the intervention agent operates.
It observes other sub-worlds but only intervenes locally.
"""

import torch
import torch.nn as nn
from typing import Optional

import sys
sys.path.append('..')
from core.subworld import SubWorld, SubWorldState


class InterventionSubWorld(SubWorld):
    """
    Intervention sub-world for the "Old Money" agent.
    
    This sub-world:
    - Maintains intervention state (resources, capabilities, constraints)
    - Tracks intervention history
    - Enforces intervention budgets and constraints
    - Provides interface for policy execution
    
    State components:
    - Resource levels (capital, influence, etc.)
    - Intervention capacity
    - Constraint satisfaction levels
    - Strategic position indicators
    """
    
    def __init__(
        self,
        global_world_model=None,
        name: str = "intervention",
        num_resources: int = 5,
        num_capacities: int = 5,
        latent_dim: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_resources = num_resources
        self.num_capacities = num_capacities
        
        # State dimension: [resources, capacities, constraints, position]
        state_dim = num_resources + num_capacities + num_resources + num_capacities
        
        super().__init__(
            name=name,
            state_dim=state_dim,
            latent_dim=latent_dim,
            global_world_model=global_world_model,
            device=device
        )
        
        # Intervention parameters
        self.resource_regeneration = nn.Parameter(
            torch.ones(num_resources, device=device) * 0.01
        )
        self.capacity_decay = nn.Parameter(
            torch.ones(num_capacities, device=device) * 0.99
        )
        
        # Dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.Tanh(),
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh()
        ).to(device)
        
        # Track intervention history
        self.intervention_history = []
    
    def initialize_state(self, batch_size: int = 1) -> SubWorldState:
        """
        Initialize intervention state.
        
        - Resources start at moderate levels
        - Capacities start at full
        - Constraints start satisfied
        - Position starts neutral
        """
        state = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Initialize resources (0 to 1 scale, start at 0.7)
        state[:, :self.num_resources] = 0.7 + torch.randn(
            batch_size, self.num_resources, device=self.device
        ) * 0.1
        state[:, :self.num_resources] = torch.clamp(
            state[:, :self.num_resources], 0.0, 1.0
        )
        
        # Initialize capacities (start at 0.8)
        cap_start = self.num_resources
        cap_end = cap_start + self.num_capacities
        state[:, cap_start:cap_end] = 0.8 + torch.randn(
            batch_size, self.num_capacities, device=self.device
        ) * 0.05
        state[:, cap_start:cap_end] = torch.clamp(
            state[:, cap_start:cap_end], 0.0, 1.0
        )
        
        # Initialize constraints (start satisfied at 0.9)
        const_start = cap_end
        const_end = const_start + self.num_resources
        state[:, const_start:const_end] = 0.9
        
        # Initialize position (start neutral at 0.5)
        pos_start = const_end
        state[:, pos_start:] = 0.5
        
        return SubWorldState(state=state, metadata={'timestep': 0})
    
    def local_dynamics(
        self,
        state: SubWorldState,
        action: Optional[torch.Tensor] = None
    ) -> SubWorldState:
        """
        Compute intervention sub-world dynamics.
        
        Dynamics include:
        - Resource regeneration
        - Capacity decay and renewal
        - Constraint satisfaction tracking
        - Strategic position evolution
        
        Args:
            state: Current intervention state
            action: Intervention action (consumes resources/capacity)
        
        Returns:
            Next intervention state
        """
        batch_size = state.state.shape[0]
        current = state.state.clone()
        
        # Extract components
        resources = current[:, :self.num_resources]
        capacities = current[:, self.num_resources:self.num_resources + self.num_capacities]
        constraints = current[:, self.num_resources + self.num_capacities:
                             self.num_resources * 2 + self.num_capacities]
        position = current[:, -self.num_capacities:]
        
        # Apply learned dynamics
        dynamics_adjustment = self.dynamics_net(current) * 0.05
        
        # Resource dynamics: regeneration + noise
        resource_regen = self.resource_regeneration.unsqueeze(0).expand(batch_size, -1)
        resource_noise = torch.randn_like(resources) * 0.01
        new_resources = resources + resource_regen + resource_noise
        new_resources = new_resources + dynamics_adjustment[:, :self.num_resources]
        
        # Capacity dynamics: decay towards baseline
        capacity_baseline = 0.8
        capacity_drift = self.capacity_decay * (capacities - capacity_baseline)
        capacity_noise = torch.randn_like(capacities) * 0.01
        new_capacities = capacities + capacity_drift + capacity_noise
        
        # Constraint dynamics: slowly return to satisfied state
        constraint_drift = 0.05 * (1.0 - constraints)
        constraint_noise = torch.randn_like(constraints) * 0.01
        new_constraints = constraints + constraint_drift + constraint_noise
        
        # Position dynamics: slow evolution
        position_noise = torch.randn_like(position) * 0.02
        new_position = position + position_noise
        
        # Apply intervention action (consumes resources and capacity)
        if action is not None:
            # Action consumes resources and capacity
            action_cost = self._compute_action_cost(action, batch_size)
            
            new_resources = new_resources - action_cost[:, :self.num_resources]
            new_capacities = new_capacities - action_cost[:, self.num_resources:
                                                          self.num_resources + self.num_capacities]
            
            # Track intervention
            self.intervention_history.append({
                'timestep': state.metadata.get('timestep', 0),
                'action': action.clone().detach(),
                'cost': action_cost.clone().detach()
            })
            
            # Update position based on action
            position_change = action_cost.mean(dim=-1, keepdim=True).expand(-1, self.num_capacities) * 0.1
            new_position = new_position + position_change
        
        # Clamp all values to valid ranges
        new_resources = torch.clamp(new_resources, min=0.0, max=1.0)
        new_capacities = torch.clamp(new_capacities, min=0.0, max=1.0)
        new_constraints = torch.clamp(new_constraints, min=0.0, max=1.0)
        new_position = torch.clamp(new_position, min=0.0, max=1.0)
        
        # Assemble new state
        new_state = torch.cat([
            new_resources,
            new_capacities,
            new_constraints,
            new_position
        ], dim=-1)
        
        # Update metadata
        new_metadata = state.metadata.copy()
        new_metadata['timestep'] = new_metadata.get('timestep', 0) + 1
        
        return SubWorldState(state=new_state, metadata=new_metadata)
    
    def _compute_action_cost(self, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Compute resource and capacity cost of an action.
        
        Args:
            action: Intervention action
            batch_size: Batch size
        
        Returns:
            Cost vector [batch, num_resources + num_capacities]
        """
        # Ensure action has correct shape
        if action.shape[0] != batch_size:
            action = action.expand(batch_size, -1)
        
        # Cost is proportional to action magnitude
        action_magnitude = torch.norm(action, dim=-1, keepdim=True)
        
        # Distribute cost across resources and capacities
        cost_dim = self.num_resources + self.num_capacities
        cost = torch.ones(batch_size, cost_dim, device=self.device)
        cost = cost * action_magnitude * 0.05  # 5% cost per unit action
        
        return cost
    
    def can_afford_action(self, action: torch.Tensor) -> bool:
        """
        Check if current resources/capacities can afford an action.
        
        Args:
            action: Proposed intervention action
        
        Returns:
            True if action is affordable
        """
        state = self.get_state()
        batch_size = state.state.shape[0]
        
        cost = self._compute_action_cost(action, batch_size)
        
        resources = state.state[:, :self.num_resources]
        capacities = state.state[:, self.num_resources:self.num_resources + self.num_capacities]
        
        resource_cost = cost[:, :self.num_resources]
        capacity_cost = cost[:, self.num_resources:]
        
        # Check if we have enough resources and capacity
        can_afford = (
            (resources >= resource_cost).all() and
            (capacities >= capacity_cost).all()
        )
        
        return can_afford.item()
    
    def get_resources(self) -> torch.Tensor:
        """Get current resource levels."""
        state = self.get_state()
        return state.state[:, :self.num_resources]
    
    def get_capacities(self) -> torch.Tensor:
        """Get current capacity levels."""
        state = self.get_state()
        return state.state[:, self.num_resources:self.num_resources + self.num_capacities]
    
    def get_intervention_budget(self) -> float:
        """
        Get current intervention budget (combined resources and capacity).
        
        Returns:
            Budget score (0 to 1)
        """
        resources = self.get_resources()
        capacities = self.get_capacities()
        
        # Budget is minimum of average resource and capacity
        budget = min(resources.mean().item(), capacities.mean().item())
        return budget

