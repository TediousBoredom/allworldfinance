"""
Causal graph representation and analysis.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class CausalGraph:
    """
    Represents causal relationships between sub-worlds.
    
    Tracks:
    - Which sub-worlds can influence others
    - Strength of causal connections
    - Intervention points
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.intervention_nodes: Set[str] = set()
    
    def add_subworld(self, name: str, is_intervention: bool = False):
        """
        Add a sub-world node to the graph.
        
        Args:
            name: Sub-world name
            is_intervention: Whether this is an intervention sub-world
        """
        self.graph.add_node(name, intervention=is_intervention)
        if is_intervention:
            self.intervention_nodes.add(name)
    
    def add_causal_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        bidirectional: bool = False
    ):
        """
        Add a causal edge between sub-worlds.
        
        Args:
            source: Source sub-world
            target: Target sub-world
            weight: Strength of causal influence
            bidirectional: If True, add edge in both directions
        """
        self.graph.add_edge(source, target)
        self.edge_weights[(source, target)] = weight
        
        if bidirectional:
            self.graph.add_edge(target, source)
            self.edge_weights[(target, source)] = weight
    
    def get_influenced_subworlds(self, source: str) -> List[str]:
        """
        Get list of sub-worlds influenced by source.
        
        Args:
            source: Source sub-world
        
        Returns:
            List of influenced sub-world names
        """
        return list(self.graph.successors(source))
    
    def get_influencing_subworlds(self, target: str) -> List[str]:
        """
        Get list of sub-worlds that influence target.
        
        Args:
            target: Target sub-world
        
        Returns:
            List of influencing sub-world names
        """
        return list(self.graph.predecessors(target))
    
    def compute_causal_paths(
        self,
        source: str,
        target: str,
        max_length: Optional[int] = None
    ) -> List[List[str]]:
        """
        Find all causal paths from source to target.
        
        Args:
            source: Source sub-world
            target: Target sub-world
            max_length: Maximum path length
        
        Returns:
            List of paths (each path is a list of node names)
        """
        if max_length is None:
            paths = list(nx.all_simple_paths(self.graph, source, target))
        else:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
        return paths
    
    def compute_intervention_reach(self, intervention_node: str) -> Dict[str, int]:
        """
        Compute how many steps it takes for intervention to reach each node.
        
        Args:
            intervention_node: Intervention sub-world name
        
        Returns:
            Dict mapping node names to shortest path length
        """
        if intervention_node not in self.graph:
            return {}
        
        lengths = nx.single_source_shortest_path_length(self.graph, intervention_node)
        return dict(lengths)
    
    def identify_critical_nodes(self) -> List[Tuple[str, float]]:
        """
        Identify critical nodes (high betweenness centrality).
        
        Returns:
            List of (node_name, centrality_score) tuples
        """
        centrality = nx.betweenness_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes
    
    def check_causal_locality(self, intervention_node: str) -> bool:
        """
        Check if intervention node respects causal locality.
        
        Causal locality means the node only influences itself or
        through explicit causal paths.
        
        Args:
            intervention_node: Intervention sub-world name
        
        Returns:
            True if causal locality is respected
        """
        # Get all nodes reachable from intervention
        reachable = nx.descendants(self.graph, intervention_node)
        reachable.add(intervention_node)
        
        # Check if there are any "backdoor" paths
        # (nodes that can reach intervention node)
        influencers = nx.ancestors(self.graph, intervention_node)
        
        # Causal locality is respected if intervention doesn't create cycles
        # and only influences through forward paths
        has_cycles = not nx.is_directed_acyclic_graph(self.graph)
        
        return not has_cycles
    
    def visualize(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize the causal graph.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Node colors
        node_colors = []
        for node in self.graph.nodes():
            if node in self.intervention_nodes:
                node_colors.append('#FF6B6B')  # Red for intervention
            else:
                node_colors.append('#4ECDC4')  # Teal for observation
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=2000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with weights
        edges = self.graph.edges()
        weights = [self.edge_weights.get((u, v), 1.0) for u, v in edges]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=[w * 3 for w in weights],
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=12,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Intervention Sub-world'),
            Patch(facecolor='#4ECDC4', label='Observed Sub-world')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title('Causal Graph Structure', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """
        Convert graph to adjacency matrix.
        
        Returns:
            Adjacency matrix
        """
        return nx.to_numpy_array(self.graph)
    
    def __repr__(self) -> str:
        return (
            f"CausalGraph("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"intervention_nodes={len(self.intervention_nodes)})"
        )


def visualize_causal_structure(
    world_model,
    agent,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    Visualize causal structure of world model with agent.
    
    Args:
        world_model: GlobalWorldModel instance
        agent: OldMoneyAgent instance
        figsize: Figure size
        save_path: Path to save figure
    """
    # Build causal graph
    graph = CausalGraph()
    
    # Add all sub-worlds
    for name, subworld in world_model.subworlds.items():
        is_intervention = (subworld == agent.intervention_subworld)
        graph.add_subworld(name, is_intervention=is_intervention)
    
    # Add causal edges
    # For now, assume each sub-world only influences itself (strict locality)
    for name in world_model.subworlds.keys():
        graph.add_causal_edge(name, name, weight=1.0)
    
    # Add observation edges (intervention observes others, but doesn't influence)
    if agent.intervention_subworld:
        intervention_name = agent.intervention_subworld.name
        for subworld in agent.observed_subworlds:
            # Dashed edge for observation (not causal influence)
            # We'll represent this by adding to graph but with low weight
            graph.add_causal_edge(subworld.name, intervention_name, weight=0.3)
    
    # Visualize
    graph.visualize(figsize=figsize, save_path=save_path)
    
    return graph

