"""
Base agent interface for world model interaction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class BaseAgent(ABC, nn.Module):
    """
    Base class for agents operating in the world model.
    
    Agents can:
    - Observe states of sub-worlds (read-only)
    - Generate interventions via policies
    - Maintain internal state and memory
    """
    
    def __init__(
        self,
        name: str,
        observation_dim: int,
        action_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        
        # Internal state
        self._internal_state: Optional[torch.Tensor] = None
        
        # Memory/history
        self.observation_history: List[torch.Tensor] = []
        self.action_history: List[torch.Tensor] = []
        self.max_history_length = 1000
        
        self.to(device)
    
    @abstractmethod
    def observe(self, world_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Observe world state and extract relevant information.
        
        Args:
            world_state: Dict mapping sub-world names to observable states
        
        Returns:
            Observation vector
        """
        pass
    
    @abstractmethod
    def policy(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Generate action based on observation.
        
        Args:
            observation: Current observation
        
        Returns:
            Action to take
        """
        pass
    
    def act(self, world_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Full perception-action cycle.
        
        Args:
            world_state: Current world state
        
        Returns:
            Action to take
        """
        # Observe
        observation = self.observe(world_state)
        
        # Store observation
        self.observation_history.append(observation.detach().clone())
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
        
        # Generate action
        action = self.policy(observation)
        
        # Store action
        self.action_history.append(action.detach().clone())
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        return action
    
    def reset(self):
        """Reset agent state and history."""
        self._internal_state = None
        self.observation_history.clear()
        self.action_history.clear()
    
    def get_observation_history(self, length: Optional[int] = None) -> List[torch.Tensor]:
        """Get observation history."""
        if length is None:
            return self.observation_history
        return self.observation_history[-length:]
    
    def get_action_history(self, length: Optional[int] = None) -> List[torch.Tensor]:
        """Get action history."""
        if length is None:
            return self.action_history
        return self.action_history[-length:]
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"obs_dim={self.observation_dim}, "
            f"action_dim={self.action_dim})"
        )

