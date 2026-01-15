"""
Social/Cultural dynamics sub-world.

Models social dynamics including:
- Reputation networks
- Information flow
- Trust dynamics
- Cultural trends
"""

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

import sys
sys.path.append('..')
from core.subworld import SubWorld, SubWorldState


class SocialSubWorld(SubWorld):
    """
    Social sub-world modeling reputation, trust, and information dynamics.
    
    State components:
    - Reputation scores (per agent/entity)
    - Trust network (pairwise trust levels)
    - Information spread (awareness levels)
    - Cultural alignment (value alignment scores)
    """
    
    def __init__(
        self,
        global_world_model=None,
        name: str = "social",
        num_entities: int = 20,
        latent_dim: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_entities = num_entities
        
        # State dimension: [reputation, trust_matrix_flat, information, culture]
        # Trust matrix is symmetric, so we only store upper triangle
        num_trust_elements = num_entities * (num_entities - 1) // 2
        state_dim = num_entities + num_trust_elements + num_entities + num_entities
        
        super().__init__(
            name=name,
            state_dim=state_dim,
            latent_dim=latent_dim,
            global_world_model=global_world_model,
            device=device
        )
        
        self.num_trust_elements = num_trust_elements
        
        # Social dynamics parameters
        self.reputation_decay = nn.Parameter(torch.tensor(0.99, device=device))
        self.trust_update_rate = nn.Parameter(torch.tensor(0.05, device=device))
        self.info_spread_rate = nn.Parameter(torch.tensor(0.3, device=device))
        
        # Network structure (adjacency matrix for social connections)
        self.adjacency = self._initialize_network()
        
        # Dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.Tanh(),
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh()
        ).to(device)
    
    def _initialize_network(self) -> torch.Tensor:
        """Initialize social network structure (small-world network)."""
        # Create a small-world network structure
        adj = torch.zeros(self.num_entities, self.num_entities, device=self.device)
        
        # Ring lattice
        k = 4  # Each node connects to k nearest neighbors
        for i in range(self.num_entities):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.num_entities
                adj[i, neighbor] = 1.0
                adj[neighbor, i] = 1.0
        
        # Random rewiring (small-world property)
        rewire_prob = 0.1
        for i in range(self.num_entities):
            for j in range(i + 1, self.num_entities):
                if adj[i, j] == 1.0 and torch.rand(1).item() < rewire_prob:
                    # Rewire
                    adj[i, j] = 0.0
                    adj[j, i] = 0.0
                    new_neighbor = torch.randint(0, self.num_entities, (1,)).item()
                    if new_neighbor != i:
                        adj[i, new_neighbor] = 1.0
                        adj[new_neighbor, i] = 1.0
        
        return adj
    
    def initialize_state(self, batch_size: int = 1) -> SubWorldState:
        """
        Initialize social state.
        
        - Reputation starts near neutral (0.5)
        - Trust starts at moderate levels for connected entities
        - Information starts at low awareness
        - Culture starts with some diversity
        """
        state = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Initialize reputation (0 to 1 scale, start near 0.5)
        state[:, :self.num_entities] = 0.5 + torch.randn(
            batch_size, self.num_entities, device=self.device
        ) * 0.1
        state[:, :self.num_entities] = torch.clamp(state[:, :self.num_entities], 0.0, 1.0)
        
        # Initialize trust (based on network structure)
        trust_start = self.num_entities
        trust_end = trust_start + self.num_trust_elements
        
        # Extract upper triangle of adjacency matrix
        trust_init = self._matrix_to_flat(self.adjacency * 0.5)
        trust_init = trust_init.unsqueeze(0).expand(batch_size, -1)
        trust_init = trust_init + torch.randn_like(trust_init) * 0.1
        state[:, trust_start:trust_end] = torch.clamp(trust_init, 0.0, 1.0)
        
        # Initialize information (low initial awareness)
        info_start = trust_end
        info_end = info_start + self.num_entities
        state[:, info_start:info_end] = torch.rand(
            batch_size, self.num_entities, device=self.device
        ) * 0.2
        
        # Initialize culture (diverse initial values)
        culture_start = info_end
        state[:, culture_start:] = torch.randn(
            batch_size, self.num_entities, device=self.device
        ) * 0.5
        
        return SubWorldState(state=state, metadata={'timestep': 0})
    
    def _matrix_to_flat(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert symmetric matrix to flat upper triangle."""
        indices = torch.triu_indices(self.num_entities, self.num_entities, offset=1)
        return matrix[indices[0], indices[1]]
    
    def _flat_to_matrix(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat upper triangle to symmetric matrix."""
        batch_size = flat.shape[0] if len(flat.shape) > 1 else 1
        if len(flat.shape) == 1:
            flat = flat.unsqueeze(0)
        
        matrix = torch.zeros(
            batch_size, self.num_entities, self.num_entities,
            device=self.device
        )
        
        indices = torch.triu_indices(self.num_entities, self.num_entities, offset=1)
        matrix[:, indices[0], indices[1]] = flat
        matrix[:, indices[1], indices[0]] = flat  # Symmetry
        
        return matrix.squeeze(0) if batch_size == 1 else matrix
    
    def local_dynamics(
        self,
        state: SubWorldState,
        action: Optional[torch.Tensor] = None
    ) -> SubWorldState:
        """
        Compute social dynamics.
        
        Dynamics include:
        - Reputation evolution (decay + events)
        - Trust updates (based on interactions)
        - Information spread (network diffusion)
        - Cultural convergence/divergence
        
        Args:
            state: Current social state
            action: Optional intervention (e.g., reputation management, information campaign)
        
        Returns:
            Next social state
        """
        batch_size = state.state.shape[0]
        current = state.state.clone()
        
        # Extract components
        reputation = current[:, :self.num_entities]
        trust_flat = current[:, self.num_entities:self.num_entities + self.num_trust_elements]
        information = current[:, self.num_entities + self.num_trust_elements:
                             self.num_entities + self.num_trust_elements + self.num_entities]
        culture = current[:, -self.num_entities:]
        
        # Apply learned dynamics
        dynamics_adjustment = self.dynamics_net(current) * 0.05
        
        # Reputation dynamics: decay towards neutral + noise
        rep_drift = self.reputation_decay * (reputation - 0.5)
        rep_noise = torch.randn_like(reputation) * 0.02
        new_reputation = torch.clamp(
            reputation + rep_drift + rep_noise + dynamics_adjustment[:, :self.num_entities],
            min=0.0,
            max=1.0
        )
        
        # Trust dynamics: influenced by reputation and network structure
        trust_matrix = self._flat_to_matrix(trust_flat)
        
        # Trust updates based on reputation similarity
        rep_diff = reputation.unsqueeze(-1) - reputation.unsqueeze(-2)  # [batch, N, N]
        trust_update = -self.trust_update_rate * rep_diff.abs()
        trust_update = trust_update * self.adjacency.unsqueeze(0)  # Only connected entities
        
        new_trust_matrix = trust_matrix + trust_update
        new_trust_matrix = new_trust_matrix + torch.randn_like(trust_matrix) * 0.01
        new_trust_matrix = torch.clamp(new_trust_matrix, min=0.0, max=1.0)
        new_trust_flat = self._matrix_to_flat(new_trust_matrix.squeeze(0) if batch_size == 1 else new_trust_matrix[0])
        if batch_size > 1:
            new_trust_flat = torch.stack([
                self._matrix_to_flat(new_trust_matrix[i])
                for i in range(batch_size)
            ])
        
        # Information dynamics: network diffusion
        # Information spreads through trusted connections
        info_matrix = information.unsqueeze(-1).expand(-1, -1, self.num_entities)
        trust_weighted = trust_matrix * self.adjacency.unsqueeze(0)
        info_spread = torch.bmm(trust_weighted, information.unsqueeze(-1)).squeeze(-1)
        info_spread = info_spread / (trust_weighted.sum(dim=-1) + 1e-6)
        
        new_information = information + self.info_spread_rate * (info_spread - information)
        new_information = new_information + torch.randn_like(information) * 0.01
        new_information = torch.clamp(new_information, min=0.0, max=1.0)
        
        # Cultural dynamics: convergence through trusted connections
        culture_matrix = culture.unsqueeze(-1).expand(-1, -1, self.num_entities)
        culture_influence = torch.bmm(trust_weighted, culture.unsqueeze(-1)).squeeze(-1)
        culture_influence = culture_influence / (trust_weighted.sum(dim=-1) + 1e-6)
        
        culture_drift = 0.05 * (culture_influence - culture)
        culture_noise = torch.randn_like(culture) * 0.05
        new_culture = culture + culture_drift + culture_noise
        
        # Apply intervention if provided
        if action is not None:
            action_effect = self._process_action(action, batch_size)
            # Intervention can affect reputation and information primarily
            new_reputation = torch.clamp(
                new_reputation + action_effect[:, :self.num_entities],
                min=0.0,
                max=1.0
            )
            info_start = self.num_entities + self.num_trust_elements
            info_end = info_start + self.num_entities
            new_information = torch.clamp(
                new_information + action_effect[:, info_start:info_end],
                min=0.0,
                max=1.0
            )
        
        # Assemble new state
        new_state = torch.cat([
            new_reputation,
            new_trust_flat,
            new_information,
            new_culture
        ], dim=-1)
        
        # Update metadata
        new_metadata = state.metadata.copy()
        new_metadata['timestep'] = new_metadata.get('timestep', 0) + 1
        
        return SubWorldState(state=new_state, metadata=new_metadata)
    
    def _process_action(self, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Process intervention action into state changes."""
        if action.shape[0] != batch_size:
            action = action.expand(batch_size, -1)
        
        if action.shape[-1] != self.state_dim:
            if action.shape[-1] < self.state_dim:
                padding = torch.zeros(
                    batch_size,
                    self.state_dim - action.shape[-1],
                    device=self.device
                )
                action = torch.cat([action, padding], dim=-1)
            else:
                action = action[:, :self.state_dim]
        
        return action * 0.01
    
    def get_reputation(self) -> torch.Tensor:
        """Get current reputation scores."""
        state = self.get_state()
        return state.state[:, :self.num_entities]
    
    def get_trust_matrix(self) -> torch.Tensor:
        """Get current trust network as matrix."""
        state = self.get_state()
        trust_flat = state.state[:, self.num_entities:self.num_entities + self.num_trust_elements]
        return self._flat_to_matrix(trust_flat)
    
    def get_information(self) -> torch.Tensor:
        """Get current information awareness levels."""
        state = self.get_state()
        info_start = self.num_entities + self.num_trust_elements
        info_end = info_start + self.num_entities
        return state.state[:, info_start:info_end]

