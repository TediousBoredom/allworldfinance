"""
Finance/Market sub-world implementation.

Models financial market dynamics including:
- Asset prices
- Market sentiment
- Trading volumes
- Volatility
"""

import torch
import torch.nn as nn
from typing import Optional
import math

import sys
sys.path.append('..')
from core.subworld import SubWorld, SubWorldState


class FinanceSubWorld(SubWorld):
    """
    Finance sub-world modeling market dynamics.
    
    State components:
    - Asset prices (log scale)
    - Price momentum
    - Volatility
    - Trading volume
    - Market sentiment
    """
    
    def __init__(
        self,
        global_world_model=None,
        name: str = "finance",
        num_assets: int = 10,
        latent_dim: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_assets = num_assets
        
        # State dimension: [prices, momentum, volatility, volume, sentiment]
        state_dim = num_assets * 5
        
        super().__init__(
            name=name,
            state_dim=state_dim,
            latent_dim=latent_dim,
            global_world_model=global_world_model,
            device=device
        )
        
        # Market dynamics parameters
        self.mean_reversion_rate = nn.Parameter(torch.ones(num_assets, device=device) * 0.1)
        self.volatility_base = nn.Parameter(torch.ones(num_assets, device=device) * 0.02)
        self.momentum_decay = nn.Parameter(torch.ones(num_assets, device=device) * 0.95)
        
        # Dynamics network (learns complex market interactions)
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.Tanh(),
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh()
        ).to(device)
    
    def initialize_state(self, batch_size: int = 1) -> SubWorldState:
        """
        Initialize market state.
        
        Prices start at log(100) = 4.605
        Other components start near zero
        """
        state = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Initialize prices (log scale, starting at ~100)
        state[:, :self.num_assets] = math.log(100.0) + torch.randn(
            batch_size, self.num_assets, device=self.device
        ) * 0.1
        
        # Initialize volatility to base level
        vol_start = self.num_assets * 2
        vol_end = self.num_assets * 3
        state[:, vol_start:vol_end] = self.volatility_base.unsqueeze(0).expand(batch_size, -1)
        
        # Initialize sentiment near neutral (0)
        sent_start = self.num_assets * 4
        state[:, sent_start:] = torch.randn(batch_size, self.num_assets, device=self.device) * 0.1
        
        return SubWorldState(state=state, metadata={'timestep': 0})
    
    def local_dynamics(
        self,
        state: SubWorldState,
        action: Optional[torch.Tensor] = None
    ) -> SubWorldState:
        """
        Compute market dynamics.
        
        Dynamics include:
        - Mean reversion in prices
        - Momentum effects
        - Stochastic volatility
        - Volume dynamics
        - Sentiment evolution
        
        Args:
            state: Current market state
            action: Optional intervention (e.g., trading action, policy change)
        
        Returns:
            Next market state
        """
        batch_size = state.state.shape[0]
        current = state.state.clone()
        
        # Extract components
        prices = current[:, :self.num_assets]
        momentum = current[:, self.num_assets:self.num_assets*2]
        volatility = current[:, self.num_assets*2:self.num_assets*3]
        volume = current[:, self.num_assets*3:self.num_assets*4]
        sentiment = current[:, self.num_assets*4:]
        
        # Apply learned dynamics
        dynamics_adjustment = self.dynamics_net(current) * 0.1
        
        # Price dynamics: log-price random walk with mean reversion and momentum
        price_drift = -self.mean_reversion_rate * (prices - math.log(100.0))
        price_drift = price_drift + momentum * 0.5
        price_noise = torch.randn_like(prices) * volatility
        new_prices = prices + price_drift + price_noise + dynamics_adjustment[:, :self.num_assets]
        
        # Momentum dynamics: decay + new shocks
        momentum_shock = torch.randn_like(momentum) * 0.01
        new_momentum = self.momentum_decay * momentum + momentum_shock
        new_momentum = new_momentum + dynamics_adjustment[:, self.num_assets:self.num_assets*2]
        
        # Volatility dynamics: stochastic volatility (mean-reverting)
        vol_target = self.volatility_base.unsqueeze(0).expand(batch_size, -1)
        vol_drift = 0.1 * (vol_target - volatility)
        vol_noise = torch.randn_like(volatility) * 0.005
        new_volatility = torch.clamp(
            volatility + vol_drift + vol_noise,
            min=0.001,
            max=0.1
        )
        
        # Volume dynamics: correlated with volatility and sentiment
        volume_drift = 0.1 * (volatility - volume)
        volume_noise = torch.randn_like(volume) * 0.02
        new_volume = torch.clamp(volume + volume_drift + volume_noise, min=0.0)
        
        # Sentiment dynamics: mean-reverting with momentum influence
        sentiment_drift = -0.05 * sentiment + 0.1 * momentum
        sentiment_noise = torch.randn_like(sentiment) * 0.05
        new_sentiment = torch.clamp(
            sentiment + sentiment_drift + sentiment_noise,
            min=-1.0,
            max=1.0
        )
        
        # Apply intervention if provided
        if action is not None:
            # Action can modify any component
            # For example, action could be a trading strategy or policy intervention
            action_effect = self._process_action(action, batch_size)
            new_prices = new_prices + action_effect[:, :self.num_assets]
            new_sentiment = torch.clamp(
                new_sentiment + action_effect[:, self.num_assets*4:],
                min=-1.0,
                max=1.0
            )
        
        # Assemble new state
        new_state = torch.cat([
            new_prices,
            new_momentum,
            new_volatility,
            new_volume,
            new_sentiment
        ], dim=-1)
        
        # Update metadata
        new_metadata = state.metadata.copy()
        new_metadata['timestep'] = new_metadata.get('timestep', 0) + 1
        
        return SubWorldState(state=new_state, metadata=new_metadata)
    
    def _process_action(self, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Process intervention action into state changes.
        
        Args:
            action: Intervention action
            batch_size: Batch size
        
        Returns:
            State change vector
        """
        # Ensure action has correct shape
        if action.shape[0] != batch_size:
            action = action.expand(batch_size, -1)
        
        # Ensure action has correct dimension
        if action.shape[-1] != self.state_dim:
            # Pad or project to correct dimension
            if action.shape[-1] < self.state_dim:
                padding = torch.zeros(
                    batch_size,
                    self.state_dim - action.shape[-1],
                    device=self.device
                )
                action = torch.cat([action, padding], dim=-1)
            else:
                action = action[:, :self.state_dim]
        
        # Scale action to reasonable magnitude
        return action * 0.01
    
    def get_prices(self) -> torch.Tensor:
        """Get current asset prices (in original scale, not log)."""
        state = self.get_state()
        log_prices = state.state[:, :self.num_assets]
        return torch.exp(log_prices)
    
    def get_returns(self, window: int = 1) -> Optional[torch.Tensor]:
        """
        Compute returns over specified window.
        
        Args:
            window: Number of timesteps for return calculation
        
        Returns:
            Returns tensor or None if insufficient history
        """
        if len(self.state_history) < window:
            return None
        
        current_prices = self.get_prices()
        past_state = self.state_history[-window]
        past_log_prices = past_state.state[:, :self.num_assets]
        past_prices = torch.exp(past_log_prices)
        
        returns = (current_prices - past_prices) / past_prices
        return returns
    
    def get_volatility(self) -> torch.Tensor:
        """Get current volatility."""
        state = self.get_state()
        return state.state[:, self.num_assets*2:self.num_assets*3]
    
    def get_sentiment(self) -> torch.Tensor:
        """Get current market sentiment."""
        state = self.get_state()
        return state.state[:, self.num_assets*4:]

