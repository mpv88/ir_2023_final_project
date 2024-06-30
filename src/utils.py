import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional

def compute_pagerank_error(exact_pagerank_scores: np.ndarray, pagerank_scores: np.ndarray, order: float = 1.0) -> np.float64:
    '''Compute the L1 norm error between exact and iterative PageRank scores'''
    error = np.linalg.norm(exact_pagerank_scores - pagerank_scores, order)
    return error

def plot_residuals(residuals: list, title: str, save_path: str = None) -> None:
    '''Plot the residuals over iterations'''
    plt.figure(figsize=(8, 6))
    plt.semilogy(residuals, marker='o', markersize=5, linestyle='-', color='b', label='Residuals')
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Log Residual', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def check_pagerank_sum(pagerank_scores: np.ndarray) -> None:
    '''Print the sum of the PageRank score vector'''
    pagerank_sum = np.sum(pagerank_scores)
    print(f"The sum of the PageRank score vector is: {pagerank_sum:.2f}")
    
def plot_network(adj_matrix: np.ndarray, page_ranks: Optional[np.ndarray] = None, max_nodes_to_show: Optional[int] = None, node_labels: Optional[pd.DataFrame] = None) -> None:
    '''Plot the network with optional PageRank scores and node labels'''
    
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Determine nodes to show based on PageRank scores or degree
    if max_nodes_to_show is not None:
        if page_ranks is not None:
            top_indices = np.argsort(page_ranks)[-max_nodes_to_show:]
        else:
            degrees = np.array([deg for _, deg in G.degree()])
            top_indices = np.argsort(degrees)[-max_nodes_to_show:]

        subgraph = G.subgraph(top_indices)
    else:
        subgraph = G

    pos = nx.spring_layout(subgraph)

    plt.figure(figsize=(5, 5))

    # Draw nodes with size proportional to PageRank scores if provided
    if page_ranks is not None:
        node_size = [5000 * page_ranks[node] for node in subgraph.nodes()]
    else:
        node_size = 300

    # Draw node labels if provided
    if node_labels is not None:
        if page_ranks is not None:
            labels = {row[0]: f"{row[1]}\n{page_ranks[row[0]]:.3%}" for row in node_labels.itertuples(index=False)}
            labels = {i: labels[i] for i in subgraph.nodes() if i in labels}
        else:
            labels = {row[0]: row[1] for row in node_labels.itertuples(index=False)}
            labels = {i: labels[i] for i in subgraph.nodes() if i in labels}
    else:
        labels = {i: i for i in subgraph.nodes()}

    nx.draw(subgraph, pos, with_labels=True, labels=labels, node_size=node_size, node_color="skyblue", edge_color="gray", font_size=10, font_color="black", font_weight="bold")

    plt.title('Network Graph')
    plt.show()