"""
Global world model that orchestrates multiple sub-worlds.

The global world model:
- Maintains overall state via diffusion
- Coordinates multiple sub-worlds
- Tracks causal relationships
- Ensures consistency across sub-worlds
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .diffusion import DiffusionModel, DDPMScheduler
from .subworld import SubWorld, SubWorldState


class GlobalWorldModel(nn.Module):
    """
    Global diffusion-based world model containing multiple sub-worlds.
    
    Architecture:
        Global State = [SubWorld_A | SubWorld_B | ... | SubWorld_N]
        
    Each sub-world operates on a slice of the global state with causal locality.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 1000,
        num_layers: int = 6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Diffusion model for global state evolution
        self.diffusion_model = DiffusionModel(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_steps=num_diffusion_steps,
            device=device
        )
        
        # Sub-worlds registry
        self.subworlds: Dict[str, SubWorld] = {}
        self.subworld_slices: Dict[str, Tuple[int, int]] = {}  # (start, end) indices
        
        # Global state
        self._global_state: Optional[torch.Tensor] = None
        self._timestep: int = 0
        
        # History
        self.global_state_history: List[torch.Tensor] = []
        self.max_history_length = 1000
        
        self.to(device)
    
    def register_subworld(
        self,
        subworld: SubWorld,
        state_slice: Optional[Tuple[int, int]] = None
    ):
        """
        Register a sub-world with the global model.
        
        Args:
            subworld: SubWorld instance to register
            state_slice: (start, end) indices in global state (auto-assigned if None)
        """
        if subworld.name in self.subworlds:
            raise ValueError(f"Sub-world {subworld.name} already registered")
        
        # Auto-assign slice if not provided
        if state_slice is None:
            if not self.subworlds:
                start = 0
            else:
                # Place after last sub-world
                last_end = max(end for _, end in self.subworld_slices.values())
                start = last_end
            end = start + subworld.state_dim
            
            if end > self.state_dim:
                raise ValueError(
                    f"Sub-world {subworld.name} (dim={subworld.state_dim}) "
                    f"exceeds global state capacity (remaining={self.state_dim - start})"
                )
            state_slice = (start, end)
        
        # Register
        self.subworlds[subworld.name] = subworld
        self.subworld_slices[subworld.name] = state_slice
        subworld.global_world_model = self
        
        print(f"Registered sub-world '{subworld.name}' at slice {state_slice}")
    
    def initialize(self, batch_size: int = 1):
        """
        Initialize global state and all sub-worlds.
        
        Args:
            batch_size: Number of parallel worlds to simulate
        """
        # Initialize global state
        self._global_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Initialize each sub-world
        for name, subworld in self.subworlds.items():
            start, end = self.subworld_slices[name]
            
            # Initialize sub-world
            subworld_state = subworld.initialize_state(batch_size)
            
            # Embed in global state
            self._global_state[:, start:end] = subworld_state.state
        
        self._timestep = 0
        self.global_state_history.clear()
    
    def step(self, interventions: Optional[Dict[str, torch.Tensor]] = None):
        """
        Advance global world model by one timestep.
        
        Args:
            interventions: Dict mapping sub-world names to intervention actions
        """
        if self._global_state is None:
            raise ValueError("World model not initialized. Call initialize() first.")
        
        interventions = interventions or {}
        
        # Step each sub-world
        for name, subworld in self.subworlds.items():
            start, end = self.subworld_slices[name]
            
            # Get intervention for this sub-world
            action = interventions.get(name, None)
            
            # Step sub-world
            new_subworld_state = subworld.step(action)
            
            # Update global state
            self._global_state[:, start:end] = new_subworld_state.state
        
        # Apply global diffusion dynamics (optional refinement)
        # This can model inter-sub-world dependencies
        self._global_state = self._apply_global_dynamics(self._global_state)
        
        # Update history
        self.global_state_history.append(self._global_state.clone())
        if len(self.global_state_history) > self.max_history_length:
            self.global_state_history.pop(0)
        
        self._timestep += 1
    
    def _apply_global_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply global-level dynamics (optional).
        
        This can model weak coupling between sub-worlds while preserving
        causal locality.
        """
        # For now, just return state unchanged
        # Can add weak diffusion-based coupling here if needed
        return state
    
    def get_subworld_state(self, subworld_name: str) -> SubWorldState:
        """Get current state of a specific sub-world."""
        if subworld_name not in self.subworlds:
            raise ValueError(f"Sub-world {subworld_name} not found")
        
        return self.subworlds[subworld_name].get_state()
    
    def get_all_observable_states(self) -> Dict[str, torch.Tensor]:
        """
        Get observable latent states of all sub-worlds.
        
        Returns:
            Dict mapping sub-world names to latent states
        """
        return {
            name: subworld.get_observable_state()
            for name, subworld in self.subworlds.items()
        }
    
    def predict_future(
        self,
        horizon: int,
        num_samples: int = 1,
        interventions: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict future global states using diffusion model.
        
        Args:
            horizon: Number of timesteps to predict
            num_samples: Number of sample trajectories
            interventions: Optional interventions to condition on
        
        Returns:
            Predicted states [num_samples, horizon, state_dim]
        """
        if self._global_state is None:
            raise ValueError("World model not initialized")
        
        batch_size = self._global_state.shape[0]
        predictions = []
        
        # Current state
        current_state = self._global_state.clone()
        
        for t in range(horizon):
            # Sample next state via diffusion
            # For simplicity, we use the diffusion model to denoise from current state
            # In practice, you might want a more sophisticated prediction model
            
            # Add small noise and denoise to get next state prediction
            noise = torch.randn_like(current_state) * 0.1
            noisy_state = current_state + noise
            
            # Denoise (simplified - in practice use full diffusion sampling)
            timesteps = torch.full(
                (batch_size * num_samples,),
                self.scheduler.num_steps // 2,
                device=self.device,
                dtype=torch.long
            )
            
            with torch.no_grad():
                noise_pred = self.diffusion_model(noisy_state, timesteps)
                next_state = noisy_state - noise_pred * 0.1
            
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions, dim=1)
    
    def compute_global_stability(self, window: int = 100) -> float:
        """
        Compute global stability metric across all sub-worlds.
        
        Args:
            window: Number of recent timesteps to consider
        
        Returns:
            Global stability score
        """
        if not self.subworlds:
            return 1.0
        
        # Average stability across all sub-worlds
        stabilities = [
            subworld.compute_stability_metric(window)
            for subworld in self.subworlds.values()
        ]
        
        return sum(stabilities) / len(stabilities)
    
    def get_causal_graph(self) -> Dict[str, List[str]]:
        """
        Get causal graph showing which sub-worlds can influence others.
        
        Returns:
            Dict mapping sub-world names to list of influenced sub-worlds
        """
        # For now, assume strict causal locality (no cross-influence)
        # Can be extended to model weak coupling
        return {name: [name] for name in self.subworlds.keys()}
    
    def reset(self, batch_size: int = 1):
        """Reset world model and all sub-worlds."""
        self.initialize(batch_size)
        for subworld in self.subworlds.values():
            subworld.reset(batch_size)
    
    def get_state(self) -> torch.Tensor:
        """Get current global state."""
        if self._global_state is None:
            raise ValueError("World model not initialized")
        return self._global_state
    
    def get_timestep(self) -> int:
        """Get current timestep."""
        return self._timestep
    
    def __repr__(self) -> str:
        subworld_info = ", ".join(
            f"{name}[{start}:{end}]"
            for name, (start, end) in self.subworld_slices.items()
        )
        return (
            f"GlobalWorldModel(state_dim={self.state_dim}, "
            f"num_subworlds={len(self.subworlds)}, "
            f"subworlds={{{subworld_info}}})"
        )

