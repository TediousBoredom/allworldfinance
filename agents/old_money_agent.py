"""
Old Money Agent: Diffusion-based intervention agent with adaptive priors.

This agent embodies the "old money" philosophy:
- Operates on a localized sub-world
- Observes other agents (read-only)
- Generates interventions via conditional diffusion
- Maintains long-term stable attractors
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('..')

from .base_agent import BaseAgent
from .priors import StructuredPrior, AdaptivePrior, StabilityPrior
from core.diffusion import ConditionalDiffusionModel, DDPMScheduler


class OldMoneyAgent(BaseAgent):
    """
    "Old Money" intervention agent.
    
    Key characteristics:
    1. Observes multiple sub-worlds (finance, social, etc.)
    2. Intervenes only on its own sub-world (causal locality)
    3. Uses conditional diffusion for policy generation
    4. Adapts structured priors based on observed states
    5. Maintains long-term stability (stable attractor)
    
    The agent's policy is:
        π(a_t | z_t^self, {z_t^other}) = Diffusion(a_t | prior(z_t^self, {z_t^other}))
    """
    
    def __init__(
        self,
        name: str = "old_money",
        intervention_subworld=None,
        observed_subworlds: Optional[List] = None,
        action_dim: int = 32,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 50,
        horizon: int = 10,
        prior_type: str = "adaptive",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.intervention_subworld = intervention_subworld
        self.observed_subworlds = observed_subworlds or []
        self.horizon = horizon
        
        # Compute observation dimension
        observation_dim = sum(sw.latent_dim for sw in self.observed_subworlds)
        if intervention_subworld:
            observation_dim += intervention_subworld.latent_dim
        
        super().__init__(
            name=name,
            observation_dim=observation_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Structured prior
        prior_dim = hidden_dim // 2
        if prior_type == "adaptive":
            self.structured_prior = AdaptivePrior(
                prior_dim=prior_dim,
                observed_dim=observation_dim,
                hidden_dim=hidden_dim,
                device=device
            )
        elif prior_type == "stability":
            self.structured_prior = StabilityPrior(
                prior_dim=prior_dim,
                observed_dim=observation_dim,
                hidden_dim=hidden_dim,
                device=device
            )
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        # Conditional diffusion model for policy
        self.policy_diffusion = ConditionalDiffusionModel(
            state_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            observed_agents_dim=observation_dim,
            prior_dim=prior_dim,
            history_dim=0,  # Can add history encoding later
            dropout=0.1
        )
        
        # Diffusion scheduler
        self.scheduler = DDPMScheduler(
            num_steps=num_diffusion_steps,
            beta_start=1e-4,
            beta_end=0.02,
            device=device
        )
        
        # Value function for action evaluation (optional)
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Stability tracker
        self.stability_history: List[float] = []
        
        self.to(device)
    
    def observe(self, world_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Observe states of all relevant sub-worlds.
        
        Args:
            world_state: Dict mapping sub-world names to observable latent states
        
        Returns:
            Concatenated observation vector
        """
        observations = []
        
        # Observe own sub-world
        if self.intervention_subworld:
            own_state = self.intervention_subworld.get_observable_state()
            observations.append(own_state)
        
        # Observe other sub-worlds (read-only)
        for subworld in self.observed_subworlds:
            other_state = subworld.get_observable_state()
            observations.append(other_state)
        
        # Concatenate all observations
        if observations:
            obs = torch.cat(observations, dim=-1)
        else:
            obs = torch.zeros(1, self.observation_dim, device=self.device)
        
        return obs
    
    def policy(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Generate intervention action via conditional diffusion.
        
        The policy is conditioned on:
        1. Current observation (observed agent states)
        2. Structured prior (adapts based on context)
        3. Historical trajectory (optional)
        
        Args:
            observation: Current observation
        
        Returns:
            Intervention action
        """
        batch_size = observation.shape[0]
        
        # Generate structured prior
        observed_states = self._parse_observation(observation)
        history = self.get_observation_history(length=self.horizon)
        prior = self.structured_prior(observed_states, history)
        
        # Generate action via conditional diffusion
        with torch.no_grad():
            action = self.policy_diffusion.sample(
                shape=(batch_size, self.action_dim),
                scheduler=self.scheduler,
                condition=None,  # Will use forward_with_prior
                num_inference_steps=self.scheduler.num_steps,
                device=self.device
            )
        
        # Alternative: use forward_with_prior for more control
        # This requires implementing a custom sampling loop
        action = self._sample_with_prior(observation, prior, batch_size)
        
        return action
    
    def _sample_with_prior(
        self,
        observation: torch.Tensor,
        prior: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Sample action using conditional diffusion with structured prior.
        
        Args:
            observation: Current observation
            prior: Structured prior
            batch_size: Batch size
        
        Returns:
            Sampled action
        """
        # Start from pure noise
        action = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Reverse diffusion with prior conditioning
        num_steps = self.scheduler.num_steps
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, device=self.device).long()
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise with prior conditioning
            with torch.no_grad():
                noise_pred = self.policy_diffusion.forward_with_prior(
                    x=action,
                    timesteps=t_batch,
                    observed_agents=observation,
                    structured_prior=prior,
                    history=None
                )
            
            # Predict x_0
            alpha_prod = self.scheduler.alphas_cumprod[t]
            beta_prod = 1 - alpha_prod
            action_0_pred = (action - beta_prod.sqrt() * noise_pred) / alpha_prod.sqrt()
            
            # Get posterior mean and variance
            posterior_mean, posterior_variance = self.scheduler.get_posterior_mean_variance(
                action, action_0_pred, t_batch
            )
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(action)
                action = posterior_mean + posterior_variance.sqrt() * noise
            else:
                action = posterior_mean
        
        return action
    
    def _parse_observation(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parse concatenated observation into sub-world components.
        
        Args:
            observation: Concatenated observation vector
        
        Returns:
            Dict mapping sub-world names to observations
        """
        observed_states = {}
        idx = 0
        
        # Own sub-world
        if self.intervention_subworld:
            dim = self.intervention_subworld.latent_dim
            observed_states[self.intervention_subworld.name] = observation[:, idx:idx+dim]
            idx += dim
        
        # Other sub-worlds
        for subworld in self.observed_subworlds:
            dim = subworld.latent_dim
            observed_states[subworld.name] = observation[:, idx:idx+dim]
            idx += dim
        
        return observed_states
    
    def evaluate_action(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate action quality using value function.
        
        Args:
            observation: Current observation
            action: Action to evaluate
        
        Returns:
            Value estimate
        """
        state_action = torch.cat([observation, action], dim=-1)
        value = self.value_net(state_action)
        return value
    
    def intervene(
        self,
        world_state: Dict[str, torch.Tensor],
        enforce_budget: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Full intervention cycle: observe, decide, act.
        
        Args:
            world_state: Current world state
            enforce_budget: If True, check intervention budget before acting
        
        Returns:
            Intervention action or None if budget insufficient
        """
        # Observe
        observation = self.observe(world_state)
        
        # Generate action
        action = self.policy(observation)
        
        # Check budget if required
        if enforce_budget and self.intervention_subworld:
            if not self.intervention_subworld.can_afford_action(action):
                # Cannot afford intervention, return zero action
                return torch.zeros_like(action)
        
        # Store in history
        self.observation_history.append(observation.detach().clone())
        self.action_history.append(action.detach().clone())
        
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        return action
    
    def compute_stability_score(self, window: int = 100) -> float:
        """
        Compute stability score based on recent observations.
        
        Measures how stable the observed world state has been.
        
        Args:
            window: Number of recent timesteps to consider
        
        Returns:
            Stability score (higher = more stable)
        """
        if len(self.observation_history) < 2:
            return 1.0
        
        recent_obs = self.get_observation_history(window)
        if len(recent_obs) < 2:
            return 1.0
        
        # Compute variance of observation changes
        obs_changes = []
        for i in range(1, len(recent_obs)):
            change = torch.norm(recent_obs[i] - recent_obs[i-1]).item()
            obs_changes.append(change)
        
        # Stability = 1 / (1 + variance)
        variance = torch.tensor(obs_changes).var().item()
        stability = 1.0 / (1.0 + variance)
        
        # Track stability
        self.stability_history.append(stability)
        if len(self.stability_history) > 1000:
            self.stability_history.pop(0)
        
        return stability
    
    def get_long_term_stability(self) -> float:
        """
        Get long-term stability metric.
        
        Returns:
            Average stability over entire history
        """
        if not self.stability_history:
            return 1.0
        return sum(self.stability_history) / len(self.stability_history)
    
    def train_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor
    ) -> Dict[str, float]:
        """
        Training step for the agent (optional).
        
        Can be used to fine-tune the policy based on outcomes.
        
        Args:
            observation: Observation
            action: Action taken
            reward: Reward received
        
        Returns:
            Dict of training metrics
        """
        # This is a placeholder for training logic
        # In practice, you might use:
        # - Diffusion policy gradient
        # - Value function learning
        # - Prior adaptation
        
        metrics = {
            'reward': reward.mean().item(),
            'action_norm': torch.norm(action).item(),
        }
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"OldMoneyAgent("
            f"name={self.name}, "
            f"obs_dim={self.observation_dim}, "
            f"action_dim={self.action_dim}, "
            f"horizon={self.horizon}, "
            f"stability={self.get_long_term_stability():.3f})"
        )

