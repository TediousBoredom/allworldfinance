"""
Metrics for evaluating stability and attractor properties.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


def compute_stability_metrics(
    state_history: List[torch.Tensor],
    window: int = 50
) -> Dict[str, float]:
    """
    Compute comprehensive stability metrics.
    
    Args:
        state_history: List of state tensors
        window: Window size for local metrics
    
    Returns:
        Dict of stability metrics
    """
    if len(state_history) < 2:
        return {'stability': 1.0}
    
    states = torch.stack(state_history).squeeze()
    
    # 1. Variance-based stability
    state_changes = []
    for i in range(1, len(states)):
        change = torch.norm(states[i] - states[i-1]).item()
        state_changes.append(change)
    
    variance = np.var(state_changes)
    variance_stability = 1.0 / (1.0 + variance)
    
    # 2. Mean reversion (tendency to return to mean)
    mean_state = states.mean(dim=0)
    distances_from_mean = [torch.norm(s - mean_state).item() for s in states]
    mean_reversion = 1.0 / (1.0 + np.std(distances_from_mean))
    
    # 3. Lyapunov-like stability (sensitivity to perturbations)
    if len(states) > window:
        lyapunov_estimates = []
        for i in range(window, len(states)):
            recent = states[i-window:i]
            perturbations = recent[1:] - recent[:-1]
            avg_perturbation = torch.norm(perturbations, dim=-1).mean().item()
            lyapunov_estimates.append(avg_perturbation)
        lyapunov_stability = 1.0 / (1.0 + np.mean(lyapunov_estimates))
    else:
        lyapunov_stability = variance_stability
    
    # 4. Autocorrelation (temporal consistency)
    if len(state_changes) > 1:
        autocorr = np.corrcoef(state_changes[:-1], state_changes[1:])[0, 1]
        autocorr_stability = (1.0 + autocorr) / 2.0  # Map [-1, 1] to [0, 1]
    else:
        autocorr_stability = 1.0
    
    # 5. Overall stability (weighted combination)
    overall_stability = (
        0.4 * variance_stability +
        0.3 * mean_reversion +
        0.2 * lyapunov_stability +
        0.1 * autocorr_stability
    )
    
    return {
        'overall_stability': overall_stability,
        'variance_stability': variance_stability,
        'mean_reversion': mean_reversion,
        'lyapunov_stability': lyapunov_stability,
        'autocorr_stability': autocorr_stability,
        'mean_change': np.mean(state_changes),
        'max_change': np.max(state_changes),
    }


def compute_attractor_metrics(
    state_history: List[torch.Tensor],
    epsilon: float = 0.1
) -> Dict[str, float]:
    """
    Compute metrics related to attractor properties.
    
    Args:
        state_history: List of state tensors
        epsilon: Threshold for considering states as "same"
    
    Returns:
        Dict of attractor metrics
    """
    if len(state_history) < 10:
        return {'attractor_strength': 0.0}
    
    states = torch.stack(state_history).squeeze().cpu().numpy()
    
    # 1. Attractor basin size (how many states are close to attractor)
    mean_state = states.mean(axis=0)
    distances = np.linalg.norm(states - mean_state, axis=1)
    basin_size = (distances < epsilon).sum() / len(distances)
    
    # 2. Attractor stability (how quickly states return to attractor)
    return_times = []
    in_basin = distances < epsilon
    outside_indices = np.where(~in_basin)[0]
    
    for idx in outside_indices:
        # Find next time state returns to basin
        future_in_basin = in_basin[idx+1:]
        if future_in_basin.any():
            return_time = np.argmax(future_in_basin) + 1
            return_times.append(return_time)
    
    if return_times:
        avg_return_time = np.mean(return_times)
        attractor_strength = 1.0 / (1.0 + avg_return_time)
    else:
        attractor_strength = basin_size
    
    # 3. Number of attractors (clustering)
    # Use simple distance-based clustering
    dist_matrix = squareform(pdist(states))
    num_attractors = estimate_num_clusters(dist_matrix, epsilon)
    
    # 4. Attractor dimensionality (effective degrees of freedom)
    # Use PCA to estimate intrinsic dimensionality
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(states)
    explained_var = pca.explained_variance_ratio_
    # Number of components explaining 95% variance
    cumsum = np.cumsum(explained_var)
    intrinsic_dim = np.argmax(cumsum >= 0.95) + 1
    
    return {
        'attractor_strength': attractor_strength,
        'basin_size': basin_size,
        'num_attractors': num_attractors,
        'intrinsic_dimensionality': intrinsic_dim,
        'avg_return_time': np.mean(return_times) if return_times else float('inf'),
    }


def estimate_num_clusters(dist_matrix: np.ndarray, epsilon: float) -> int:
    """
    Estimate number of clusters using distance matrix.
    
    Args:
        dist_matrix: Pairwise distance matrix
        epsilon: Distance threshold
    
    Returns:
        Estimated number of clusters
    """
    n = len(dist_matrix)
    visited = np.zeros(n, dtype=bool)
    num_clusters = 0
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Start new cluster
        num_clusters += 1
        cluster = [i]
        visited[i] = True
        
        # Add all points within epsilon
        for j in range(n):
            if not visited[j] and dist_matrix[i, j] < epsilon:
                cluster.append(j)
                visited[j] = True
    
    return num_clusters


def compute_intervention_efficiency(
    action_history: List[torch.Tensor],
    stability_history: List[float]
) -> Dict[str, float]:
    """
    Compute efficiency of interventions.
    
    Measures how much stability is achieved per unit of intervention.
    
    Args:
        action_history: List of action tensors
        stability_history: List of stability scores
    
    Returns:
        Dict of efficiency metrics
    """
    if not action_history or not stability_history:
        return {'efficiency': 0.0}
    
    # Compute total intervention cost
    action_norms = [torch.norm(a).item() for a in action_history]
    total_cost = sum(action_norms)
    avg_cost = np.mean(action_norms)
    
    # Compute stability benefit
    avg_stability = np.mean(stability_history)
    stability_improvement = stability_history[-1] - stability_history[0] if len(stability_history) > 1 else 0.0
    
    # Efficiency = stability / cost
    efficiency = avg_stability / (avg_cost + 1e-6)
    improvement_efficiency = stability_improvement / (total_cost + 1e-6)
    
    return {
        'efficiency': efficiency,
        'improvement_efficiency': improvement_efficiency,
        'total_cost': total_cost,
        'avg_cost': avg_cost,
        'avg_stability': avg_stability,
        'stability_improvement': stability_improvement,
    }


def compute_causal_impact(
    intervention_states: List[torch.Tensor],
    observed_states: Dict[str, List[torch.Tensor]],
    lag: int = 1
) -> Dict[str, float]:
    """
    Compute causal impact of interventions on observed sub-worlds.
    
    Args:
        intervention_states: States of intervention sub-world
        observed_states: Dict mapping sub-world names to state histories
        lag: Time lag for causal effect
    
    Returns:
        Dict mapping sub-world names to causal impact scores
    """
    if len(intervention_states) < lag + 2:
        return {}
    
    impacts = {}
    
    # Compute intervention changes
    intervention_changes = []
    for i in range(1, len(intervention_states)):
        change = torch.norm(intervention_states[i] - intervention_states[i-1]).item()
        intervention_changes.append(change)
    
    # Compute correlation with observed changes (with lag)
    for name, states in observed_states.items():
        if len(states) < lag + 2:
            continue
        
        observed_changes = []
        for i in range(1, len(states)):
            change = torch.norm(states[i] - states[i-1]).item()
            observed_changes.append(change)
        
        # Compute lagged correlation
        if len(intervention_changes) > lag and len(observed_changes) > lag:
            intervention_lagged = intervention_changes[:-lag] if lag > 0 else intervention_changes
            observed_lagged = observed_changes[lag:]
            
            min_len = min(len(intervention_lagged), len(observed_lagged))
            if min_len > 1:
                correlation = np.corrcoef(
                    intervention_lagged[:min_len],
                    observed_lagged[:min_len]
                )[0, 1]
                impacts[name] = abs(correlation)  # Absolute correlation as impact
    
    return impacts


def compute_long_term_stability_score(
    state_history: List[torch.Tensor],
    windows: List[int] = [10, 50, 100]
) -> Dict[str, float]:
    """
    Compute stability at multiple time scales.
    
    Args:
        state_history: List of state tensors
        windows: List of window sizes
    
    Returns:
        Dict of multi-scale stability scores
    """
    scores = {}
    
    for window in windows:
        if len(state_history) < window:
            continue
        
        metrics = compute_stability_metrics(state_history[-window:], window=window//2)
        scores[f'stability_window_{window}'] = metrics['overall_stability']
    
    # Compute trend (is stability improving?)
    if len(state_history) >= max(windows):
        early_stability = compute_stability_metrics(
            state_history[:len(state_history)//2]
        )['overall_stability']
        late_stability = compute_stability_metrics(
            state_history[len(state_history)//2:]
        )['overall_stability']
        scores['stability_trend'] = late_stability - early_stability
    
    return scores

