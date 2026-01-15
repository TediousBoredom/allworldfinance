"""
Sub-world base class and state representation.

Sub-worlds are localized regions within the global world model with:
- Local state space
- Local dynamics
- Causal boundaries (interventions confined to sub-world)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class SubWorldState:
    """
    State representation for a sub-world.
    
    Attributes:
        state: Current state vector
        latent: Latent representation (for observation by other agents)
        metadata: Additional metadata (e.g., timestamps, labels)
    """
    state: torch.Tensor
    latent: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: str):
        """Move state to device."""
        self.state = self.state.to(device)
        if self.latent is not None:
            self.latent = self.latent.to(device)
        return self
    
    def clone(self):
        """Create a deep copy of the state."""
        return SubWorldState(
            state=self.state.clone(),
            latent=self.latent.clone() if self.latent is not None else None,
            metadata=self.metadata.copy()
        )


class SubWorld(ABC, nn.Module):
    """
    Base class for sub-worlds within the global world model.
    
    Each sub-world:
    - Has a local state space (subset of global state)
    - Defines local dynamics
    - Can be observed by other agents (via latent representation)
    - Enforces causal locality (interventions only affect local state)
    """
    
    def __init__(
        self,
        name: str,
        state_dim: int,
        latent_dim: int,
        global_world_model: Optional['GlobalWorldModel'] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = device
        self.global_world_model = global_world_model
        
        # State encoder: maps state to latent (for observation by other agents)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Current state
        self._current_state: Optional[SubWorldState] = None
        
        # History
        self.state_history: List[SubWorldState] = []
        self.max_history_length = 1000
        
        self.to(device)
    
    @abstractmethod
    def initialize_state(self, batch_size: int = 1) -> SubWorldState:
        """
        Initialize the sub-world state.
        
        Args:
            batch_size: Number of parallel states to initialize
        
        Returns:
            Initial state
        """
        pass
    
    @abstractmethod
    def local_dynamics(
        self,
        state: SubWorldState,
        action: Optional[torch.Tensor] = None
    ) -> SubWorldState:
        """
        Compute local dynamics: s_{t+1} = f(s_t, a_t)
        
        Args:
            state: Current state
            action: Optional action/intervention
        
        Returns:
            Next state
        """
        pass
    
    def encode_state(self, state: SubWorldState) -> torch.Tensor:
        """
        Encode state to latent representation for observation by other agents.
        
        Args:
            state: State to encode
        
        Returns:
            Latent representation
        """
        return self.state_encoder(state.state)
    
    def get_observable_state(self) -> torch.Tensor:
        """
        Get observable latent state for other agents (read-only).
        
        Returns:
            Latent representation of current state
        """
        if self._current_state is None:
            raise ValueError(f"Sub-world {self.name} has not been initialized")
        
        if self._current_state.latent is None:
            self._current_state.latent = self.encode_state(self._current_state)
        
        return self._current_state.latent.detach()  # Read-only
    
    def apply_intervention(
        self,
        action: torch.Tensor,
        enforce_causality: bool = True
    ) -> SubWorldState:
        """
        Apply intervention to local sub-world.
        
        Args:
            action: Intervention action
            enforce_causality: If True, ensure action only affects local state
        
        Returns:
            New state after intervention
        """
        if self._current_state is None:
            raise ValueError(f"Sub-world {self.name} has not been initialized")
        
        # Apply local dynamics with intervention
        new_state = self.local_dynamics(self._current_state, action)
        
        # Enforce causal locality
        if enforce_causality:
            new_state = self._enforce_causal_boundary(new_state)
        
        return new_state
    
    def step(self, action: Optional[torch.Tensor] = None) -> SubWorldState:
        """
        Advance sub-world by one timestep.
        
        Args:
            action: Optional intervention action
        
        Returns:
            New state
        """
        if self._current_state is None:
            self._current_state = self.initialize_state()
        
        # Compute next state
        new_state = self.local_dynamics(self._current_state, action)
        
        # Update current state
        self._update_state(new_state)
        
        return self._current_state
    
    def _update_state(self, new_state: SubWorldState):
        """Update current state and history."""
        # Add to history
        if self._current_state is not None:
            self.state_history.append(self._current_state.clone())
            if len(self.state_history) > self.max_history_length:
                self.state_history.pop(0)
        
        # Update current state
        self._current_state = new_state
        self._current_state.latent = self.encode_state(new_state)
    
    def _enforce_causal_boundary(self, state: SubWorldState) -> SubWorldState:
        """
        Enforce causal boundary: ensure state changes are confined to sub-world.
        
        This is a placeholder - subclasses can override for specific constraints.
        """
        return state
    
    def get_state(self) -> SubWorldState:
        """Get current state."""
        if self._current_state is None:
            raise ValueError(f"Sub-world {self.name} has not been initialized")
        return self._current_state
    
    def set_state(self, state: SubWorldState):
        """Set current state."""
        self._current_state = state
        self._current_state.latent = self.encode_state(state)
    
    def reset(self, batch_size: int = 1):
        """Reset sub-world to initial state."""
        self._current_state = self.initialize_state(batch_size)
        self.state_history.clear()
    
    def get_history(self, length: Optional[int] = None) -> List[SubWorldState]:
        """
        Get state history.
        
        Args:
            length: Number of recent states to return (None = all)
        
        Returns:
            List of historical states
        """
        if length is None:
            return self.state_history
        return self.state_history[-length:]
    
    def compute_stability_metric(self, window: int = 100) -> float:
        """
        Compute stability metric based on recent state history.
        
        Measures how stable the sub-world state has been.
        
        Args:
            window: Number of recent timesteps to consider
        
        Returns:
            Stability score (higher = more stable)
        """
        if len(self.state_history) < 2:
            return 1.0
        
        recent_states = self.get_history(window)
        if len(recent_states) < 2:
            return 1.0
        
        # Compute variance of state changes
        state_changes = []
        for i in range(1, len(recent_states)):
            change = torch.norm(
                recent_states[i].state - recent_states[i-1].state
            ).item()
            state_changes.append(change)
        
        # Stability = 1 / (1 + variance)
        variance = torch.tensor(state_changes).var().item()
        stability = 1.0 / (1.0 + variance)
        
        return stability
    
    def get_causal_influence_mask(self) -> torch.Tensor:
        """
        Get mask indicating which dimensions of global state this sub-world can influence.
        
        Returns:
            Binary mask [global_state_dim]
        """
        # Default: sub-world can only influence its own dimensions
        # Subclasses can override for more complex causal structures
        if self.global_world_model is None:
            return torch.ones(self.state_dim, device=self.device)
        
        global_dim = self.global_world_model.state_dim
        mask = torch.zeros(global_dim, device=self.device)
        # This is a placeholder - actual implementation depends on how
        # sub-world states are embedded in global state
        return mask
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"state_dim={self.state_dim}, "
            f"latent_dim={self.latent_dim})"
        )

