# Old Money: Diffusion-Based Hierarchical World Model

A sophisticated implementation of a diffusion-based world model with hierarchical sub-worlds and adaptive agent interventions, embodying the "old money" philosophy: **long-term stable attractor states through localized causal interventions**.

## Core Philosophy

"Old money" is not about wealth scale, but about maintaining a **stable attractor in the world model** through:
- **Localized sub-world interventions**: Agents operate within bounded causal domains
- **Adaptive structured priors**: Policies dynamically adjust based on observed agent states
- **Long-term equilibrium**: Strategic generation of stable states rather than reactive optimization

## Architecture

```
Global World Model (Diffusion-based)
│
├── Sub-world A: Finance/Market Dynamics
│   ├── Price evolution
│   ├── Market sentiment
│   └── Trading volumes
│
├── Sub-world B: Social/Cultural Dynamics
│   ├── Reputation networks
│   ├── Information flow
│   └── Trust dynamics
│
└── Sub-world C: Environment Intervention Agent
    ├── Observes other agent states (read-only)
    ├── Diffusion-based future sampling
    ├── Prior-modulated local intervention
    └── Maintains causal locality
```

## Key Features

1. **Hierarchical Diffusion Model**: Global world model with embedded sub-worlds
2. **Causal Locality**: Strict boundaries on intervention domains
3. **Adaptive Priors**: Structured priors that condition on latent states of other agents
4. **Stable Attractors**: Long-term equilibrium maintenance through strategic interventions
5. **Interpretability**: Clear causal chains and intervention tracking

## Theoretical Foundation

The intervention policy is conditioned on the latent states of other agents:

```
π(a_t | z_t^self, {z_t^other}) = Diffusion(a_t | prior(z_t^self, {z_t^other}))
```

Where:
- `z_t^self`: Latent state of the intervention agent's sub-world
- `{z_t^other}`: Observed latent states of other agents (read-only)
- `prior(·)`: Structured prior that adapts based on social/environmental context

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from world_model import GlobalWorldModel
from subworlds import FinanceSubWorld, SocialSubWorld, InterventionSubWorld
from agents import OldMoneyAgent

# Initialize global world model
world = GlobalWorldModel(
    state_dim=128,
    latent_dim=64,
    num_diffusion_steps=1000
)

# Create sub-worlds
finance = FinanceSubWorld(world, name="finance")
social = SocialSubWorld(world, name="social")
intervention = InterventionSubWorld(world, name="intervention")

# Create old money agent
agent = OldMoneyAgent(
    intervention_subworld=intervention,
    observed_subworlds=[finance, social],
    horizon=100
)

# Run simulation
for t in range(1000):
    # Observe other agents
    observations = agent.observe()
    
    # Generate intervention via conditional diffusion
    action = agent.intervene(observations)
    
    # Apply to local sub-world only
    intervention.apply_action(action)
    
    # Evolve global world model
    world.step()
```

## Project Structure

```
allworldfinance/
├── core/
│   ├── diffusion.py          # Core diffusion model implementation
│   ├── world_model.py         # Global world model
│   └── subworld.py            # Sub-world base class
├── subworlds/
│   ├── finance.py             # Finance/market sub-world
│   ├── social.py              # Social dynamics sub-world
│   └── intervention.py        # Intervention agent sub-world
├── agents/
│   ├── base_agent.py          # Base agent interface
│   ├── old_money_agent.py     # Old money intervention agent
│   └── priors.py              # Structured prior implementations
├── utils/
│   ├── visualization.py       # Visualization tools
│   ├── metrics.py             # Stability and attractor metrics
│   └── causal_graph.py        # Causal structure tracking
├── experiments/
│   ├── stable_attractor.py    # Long-term stability experiments
│   ├── intervention_demo.py   # Intervention demonstrations
│   └── multi_agent.py         # Multi-agent scenarios
└── configs/
    └── default.yaml           # Default configuration
```

## Key Concepts

### 1. Diffusion-Based World Model

The global world model uses denoising diffusion to model state transitions:

```
p(x_{t+1} | x_t) = ∫ p(x_{t+1} | z) p(z | x_t) dz
```

### 2. Sub-World Causal Locality

Each sub-world has:
- **Local state space**: `S_i ⊂ S_global`
- **Local dynamics**: `f_i: S_i → S_i`
- **Causal boundary**: Interventions cannot affect states outside `S_i`

### 3. Adaptive Structured Priors

Priors adapt based on observed agent states:

```
prior_t = g(z_t^self, {z_t^other}, history)
```

This allows the agent to:
- Anticipate market shifts
- Adjust to changing social contexts
- Maintain stability under perturbations

## Examples

See `experiments/` for detailed examples:

- **stable_attractor.py**: Demonstrates long-term stability maintenance
- **intervention_demo.py**: Shows adaptive intervention strategies
- **multi_agent.py**: Multi-agent coordination and competition

## Citation

```bibtex
@software{oldmoney2026,
  title={Old Money: Diffusion-Based Hierarchical World Model},
  author={Your Name},
  year={2026},
  description={A diffusion-based world model with localized sub-worlds and adaptive interventions}
}
```

## License

MIT License

