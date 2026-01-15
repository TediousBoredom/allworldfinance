"""
Old Money: Diffusion-Based Hierarchical World Model

A sophisticated implementation of a diffusion-based world model with hierarchical 
sub-worlds and adaptive agent interventions.
"""

__version__ = "0.1.0"
__author__ = "Old Money Research Team"

from core.world_model import GlobalWorldModel
from core.diffusion import DiffusionModel, DDPMScheduler, ConditionalDiffusionModel
from core.subworld import SubWorld, SubWorldState

from subworlds.finance import FinanceSubWorld
from subworlds.social import SocialSubWorld
from subworlds.intervention import InterventionSubWorld

from agents.old_money_agent import OldMoneyAgent
from agents.base_agent import BaseAgent
from agents.priors import (
    StructuredPrior,
    AdaptivePrior,
    StabilityPrior,
    MultiScalePrior,
    ComposedPrior
)

__all__ = [
    # Core
    'GlobalWorldModel',
    'DiffusionModel',
    'DDPMScheduler',
    'ConditionalDiffusionModel',
    'SubWorld',
    'SubWorldState',
    
    # Sub-worlds
    'FinanceSubWorld',
    'SocialSubWorld',
    'InterventionSubWorld',
    
    # Agents
    'OldMoneyAgent',
    'BaseAgent',
    
    # Priors
    'StructuredPrior',
    'AdaptivePrior',
    'StabilityPrior',
    'MultiScalePrior',
    'ComposedPrior',
]

