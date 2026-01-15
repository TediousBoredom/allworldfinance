"""
Structured prior implementations for adaptive intervention policies.

Priors adapt based on observed agent states and environmental context,
enabling the "old money" agent to maintain stable attractors.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import torch.nn.functional as F


class StructuredPrior(nn.Module):
    """
    Base class for structured priors that condition intervention policies.
    
    Priors encode domain knowledge and adapt based on context:
    - Observed states of other agents
    - Historical trajectories
    - Environmental conditions
    """
    
    def __init__(
        self,
        prior_dim: int,
        observed_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.prior_dim = prior_dim
        self.observed_dim = observed_dim
        self.device = device
        
        self.to(device)
    
    def forward(
        self,
        observed_states: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute structured prior based on observations.
        
        Args:
            observed_states: Dict of observed agent states
            history: Optional historical trajectory
        
        Returns:
            Prior vector [batch, prior_dim]
        """
        raise NotImplementedError


class AdaptivePrior(StructuredPrior):
    """
    Adaptive prior that adjusts based on observed agent states.
    
    Key idea: The prior adapts to maintain stability by anticipating
    changes in other agents' behaviors.
    """
    
    def __init__(
        self,
        prior_dim: int,
        observed_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(prior_dim, observed_dim, device)
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # History encoder (if provided)
        self.history_encoder = nn.GRU(
            input_size=observed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Prior generator
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ])
        layers.append(nn.Linear(hidden_dim, prior_dim))
        
        self.prior_net = nn.Sequential(*layers)
        
        self.to(device)
    
    def forward(
        self,
        observed_states: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate adaptive prior.
        
        The prior adapts to:
        1. Current observed states (immediate context)
        2. Historical trajectory (temporal patterns)
        """
        # Concatenate all observed states
        obs_list = [state for state in observed_states.values()]
        if not obs_list:
            # No observations, return neutral prior
            batch_size = 1
            return torch.zeros(batch_size, self.prior_dim, device=self.device)
        
        obs_concat = torch.cat(obs_list, dim=-1)
        batch_size = obs_concat.shape[0]
        
        # Encode current observations
        obs_encoded = self.obs_encoder(obs_concat)
        
        # Encode history if provided
        if history is not None and len(history) > 0:
            history_tensor = torch.stack(history, dim=1)  # [batch, seq_len, dim]
            _, history_encoded = self.history_encoder(history_tensor)
            history_encoded = history_encoded[-1]  # Last layer hidden state
        else:
            history_encoded = torch.zeros(batch_size, obs_encoded.shape[-1], device=self.device)
        
        # Combine current and historical context
        combined = torch.cat([obs_encoded, history_encoded], dim=-1)
        
        # Generate prior
        prior = self.prior_net(combined)
        
        return prior


class StabilityPrior(StructuredPrior):
    """
    Stability-focused prior that encourages long-term equilibrium.
    
    This prior:
    - Detects deviations from stable attractors
    - Generates corrective biases
    - Adapts to changing equilibrium points
    """
    
    def __init__(
        self,
        prior_dim: int,
        observed_dim: int,
        hidden_dim: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(prior_dim, observed_dim, device)
        
        # Attractor detector: identifies stable states
        self.attractor_net = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Deviation detector: measures distance from attractor
        self.deviation_net = nn.Sequential(
            nn.Linear(observed_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Prior generator: creates stabilizing bias
        self.prior_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, prior_dim),
            nn.Tanh()
        )
        
        # Learnable attractor reference (updated via EMA)
        self.register_buffer('attractor_reference', torch.zeros(hidden_dim // 4))
        self.ema_rate = 0.99
        
        self.to(device)
    
    def forward(
        self,
        observed_states: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate stability-focused prior.
        
        The prior encourages actions that:
        1. Reduce deviation from stable attractors
        2. Maintain long-term equilibrium
        3. Adapt to shifting equilibrium points
        """
        # Concatenate observations
        obs_list = [state for state in observed_states.values()]
        if not obs_list:
            batch_size = 1
            return torch.zeros(batch_size, self.prior_dim, device=self.device)
        
        obs_concat = torch.cat(obs_list, dim=-1)
        batch_size = obs_concat.shape[0]
        
        # Detect current attractor state
        attractor_state = self.attractor_net(obs_concat)
        
        # Update attractor reference (EMA)
        if self.training:
            with torch.no_grad():
                self.attractor_reference = (
                    self.ema_rate * self.attractor_reference +
                    (1 - self.ema_rate) * attractor_state.mean(dim=0)
                )
        
        # Compute deviation from reference attractor
        attractor_ref_expanded = self.attractor_reference.unsqueeze(0).expand(batch_size, -1)
        deviation_input = torch.cat([obs_concat, attractor_ref_expanded], dim=-1)
        deviation = self.deviation_net(deviation_input)
        
        # Generate stabilizing prior
        prior = self.prior_generator(deviation)
        
        return prior


class MultiScalePrior(StructuredPrior):
    """
    Multi-scale prior that operates at different time horizons.
    
    Combines:
    - Short-term tactical adjustments
    - Medium-term strategic positioning
    - Long-term stability maintenance
    """
    
    def __init__(
        self,
        prior_dim: int,
        observed_dim: int,
        hidden_dim: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(prior_dim, observed_dim, device)
        
        # Short-term prior (reactive)
        self.short_term_net = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, prior_dim // 3)
        )
        
        # Medium-term prior (strategic)
        self.medium_term_net = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prior_dim // 3)
        )
        
        # Long-term prior (stability)
        self.long_term_net = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prior_dim - 2 * (prior_dim // 3))
        )
        
        # Temporal weighting (learned)
        self.temporal_weights = nn.Parameter(torch.ones(3, device=device) / 3)
        
        self.to(device)
    
    def forward(
        self,
        observed_states: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Generate multi-scale prior."""
        obs_list = [state for state in observed_states.values()]
        if not obs_list:
            batch_size = 1
            return torch.zeros(batch_size, self.prior_dim, device=self.device)
        
        obs_concat = torch.cat(obs_list, dim=-1)
        
        # Generate priors at different scales
        short_prior = self.short_term_net(obs_concat)
        medium_prior = self.medium_term_net(obs_concat)
        long_prior = self.long_term_net(obs_concat)
        
        # Combine with learned weights
        weights = F.softmax(self.temporal_weights, dim=0)
        
        # Concatenate and weight
        combined_prior = torch.cat([
            short_prior * weights[0],
            medium_prior * weights[1],
            long_prior * weights[2]
        ], dim=-1)
        
        return combined_prior


class ComposedPrior(StructuredPrior):
    """
    Composed prior that combines multiple prior types.
    
    Allows flexible composition of different prior strategies.
    """
    
    def __init__(
        self,
        priors: List[StructuredPrior],
        prior_dim: int,
        observed_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(prior_dim, observed_dim, device)
        
        self.priors = nn.ModuleList(priors)
        
        # Composition network
        total_prior_dim = sum(p.prior_dim for p in priors)
        self.composition_net = nn.Sequential(
            nn.Linear(total_prior_dim, prior_dim * 2),
            nn.LayerNorm(prior_dim * 2),
            nn.SiLU(),
            nn.Linear(prior_dim * 2, prior_dim)
        )
        
        self.to(device)
    
    def forward(
        self,
        observed_states: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Generate composed prior."""
        # Generate all component priors
        component_priors = [
            prior(observed_states, history)
            for prior in self.priors
        ]
        
        # Concatenate
        combined = torch.cat(component_priors, dim=-1)
        
        # Compose
        final_prior = self.composition_net(combined)
        
        return final_prior

