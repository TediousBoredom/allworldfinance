"""
Diffusion model implementation for world state evolution.

This module implements the core denoising diffusion probabilistic model (DDPM)
used for modeling state transitions in the world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class DDPMScheduler:
    """
    DDPM noise scheduler for diffusion process.
    Implements linear beta schedule for forward and reverse diffusion.
    """
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_steps = num_steps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
    
    def get_posterior_mean_variance(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for reverse diffusion.
        """
        coef1 = self.posterior_mean_coef1[timesteps]
        coef2 = self.posterior_mean_coef2[timesteps]
        variance = self.posterior_variance[timesteps]
        
        # Reshape for broadcasting
        while len(coef1.shape) < len(x_t.shape):
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
            variance = variance.unsqueeze(-1)
        
        posterior_mean = coef1 * x_0_pred + coef2 * x_t
        return posterior_mean, variance


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)
        h = self.block2(h)
        return x + h


class DiffusionModel(nn.Module):
    """
    Denoising diffusion model for state prediction.
    
    Predicts noise added to state at timestep t, conditioned on:
    - Current noisy state
    - Timestep
    - Optional conditioning (e.g., other agent states, structured priors)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        condition_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim or 0
        
        # Time embedding
        time_dim = hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        input_dim = state_dim + self.condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise in state x at timestep t.
        
        Args:
            x: Noisy state [batch, state_dim]
            timesteps: Diffusion timesteps [batch]
            condition: Optional conditioning [batch, condition_dim]
        
        Returns:
            Predicted noise [batch, state_dim]
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Concatenate condition if provided
        if condition is not None:
            x = torch.cat([x, condition], dim=-1)
        
        # Input projection
        h = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            h = block(h, time_emb)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        scheduler: DDPMScheduler,
        condition: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion process.
        
        Args:
            shape: Shape of samples to generate
            scheduler: DDPM scheduler
            condition: Optional conditioning
            num_inference_steps: Number of denoising steps (default: scheduler.num_steps)
            device: Device to generate on
        
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        if num_inference_steps is None:
            num_inference_steps = scheduler.num_steps
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        timesteps = torch.linspace(
            scheduler.num_steps - 1, 0, num_inference_steps, device=device
        ).long()
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self(x, t_batch, condition)
            
            # Predict x_0
            alpha_prod = scheduler.alphas_cumprod[t]
            beta_prod = 1 - alpha_prod
            x_0_pred = (x - beta_prod.sqrt() * noise_pred) / alpha_prod.sqrt()
            
            # Get posterior mean and variance
            posterior_mean, posterior_variance = scheduler.get_posterior_mean_variance(
                x, x_0_pred, t_batch
            )
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + posterior_variance.sqrt() * noise
            else:
                x = posterior_mean
        
        return x


class ConditionalDiffusionModel(DiffusionModel):
    """
    Extended diffusion model with structured prior conditioning.
    
    This model conditions on:
    1. Observed states of other agents
    2. Structured priors that adapt based on context
    3. Historical trajectory information
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        observed_agents_dim: int = 0,
        prior_dim: int = 0,
        history_dim: int = 0,
        dropout: float = 0.1
    ):
        condition_dim = observed_agents_dim + prior_dim + history_dim
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            condition_dim=condition_dim,
            dropout=dropout
        )
        
        self.observed_agents_dim = observed_agents_dim
        self.prior_dim = prior_dim
        self.history_dim = history_dim
    
    def forward_with_prior(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        observed_agents: Optional[torch.Tensor] = None,
        structured_prior: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with structured conditioning.
        
        Args:
            x: Noisy state
            timesteps: Diffusion timesteps
            observed_agents: States of other agents (read-only)
            structured_prior: Adaptive structured prior
            history: Historical trajectory information
        
        Returns:
            Predicted noise
        """
        # Concatenate all conditioning
        conditions = []
        if observed_agents is not None:
            conditions.append(observed_agents)
        if structured_prior is not None:
            conditions.append(structured_prior)
        if history is not None:
            conditions.append(history)
        
        condition = torch.cat(conditions, dim=-1) if conditions else None
        
        return self.forward(x, timesteps, condition)

