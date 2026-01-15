# Changelog

All notable changes to the Old Money project will be documented in this file.

## [0.1.0] - 2026-01-15

### init

基于diffusion实现，“old money”的核心不是财富规模，而是世界模型长期处于稳定吸引子：子世界 +  agent 约束的 长期稳态策略生成器，具体理论：The intervention policy is conditioned on the latent states of other agents, allowing adaptive adjustment of the structured prior under changing social and environmental contexts.,diffusion-based agent operates on a localized sub-world embedded within a larger world model, and its interventions are strictly confined to the causal dynamics of that sub-world.embed an environment intervention agent into a localized sub-world within a larger diffusion-based world model.

The agent observes projected states of other agents but only intervenes on the causal dynamics of its own sub-world.

Its policy is generated via conditional diffusion, modulated by structured priors that adapt dynamically to the surrounding agent states.

This design ensures both interpretability and strict causal locality. 框架： Global World Model (Diffusion)

│

├── Sub-world A (Finance / Market)

├── Sub-world B (other xxx)

└── Sub-world C (Environment Intervention / "Agent")

        ├── observes other agent states (read-only)

        ├── diffusion-based future sampling

        └── prior-modulated local intervention实现上述idea并扩展为完整项目

### Added
- Initial release of Old Money framework
- Core diffusion-based world model implementation
- Hierarchical sub-world architecture with causal locality
- Finance sub-world with market dynamics
- Social sub-world with reputation and trust networks
- Intervention sub-world for agent operations
- Old Money agent with adaptive structured priors
- Multiple prior strategies (Adaptive, Stability, Multi-scale, Composed)
- Comprehensive visualization tools
- Stability and attractor metrics
- Causal graph analysis
- Three demonstration experiments:
  - Stable Attractor: Long-term stability maintenance
  - Intervention Demo: Adaptive response to shocks
  - Multi-Agent: Multiple competing/cooperating agents
- Quick start example and documentation
- Configuration system with YAML support
- Command-line interface for running experiments

### Features
- Diffusion-based state evolution
- Conditional policy generation with structured priors
- Read-only observation of other sub-worlds
- Strict causal locality enforcement
- Resource and capacity management
- Intervention budget constraints
- Multi-scale temporal dynamics
- Attractor detection and analysis
- Interactive visualizations with Plotly
- Comprehensive metrics and analysis tools

### Documentation
- Complete README with architecture overview
- API documentation in docstrings
- Quick start guide
- Example experiments
- Configuration guide
- Contributing guidelines

## [Unreleased]

### Planned Features
- Training pipeline for policy optimization
- Additional sub-world implementations (e.g., environmental, technological)
- Advanced prior composition strategies
- Real-time visualization dashboard
- Distributed simulation support
- Integration with external data sources
- Benchmark suite for stability metrics

