import numpy as np
import pandas as pd
from typing import List, Tuple

class DensePageRank:
    def __init__(self):
        '''Initialize the DensePageRank class'''
        self.adjacency_matrix: np.ndarray = None
        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.node_labels: pd.DataFrame = None
        self.normalized_adjacency: np.ndarray = None
        self.normalized_adjacency_transpose: np.ndarray = None
        self.out_degrees: np.ndarray = None

    def load_network(self, dataset_path: str, is_undirected: bool = False) -> None:
        '''Load the graph dataset from the given path'''
        edges = self._load_edges(dataset_path)
        self._create_adjacency_matrix(edges, is_undirected)
        self.num_nodes = self.adjacency_matrix.shape[0]
        self.num_edges = np.count_nonzero(self.adjacency_matrix)

    def _load_edges(self, dataset_path: str) -> np.ndarray:
        '''Load edges from the dataset path'''
        edge_file_path = f"{dataset_path}/edges.tsv"
        return np.loadtxt(edge_file_path, dtype=int)

    def _create_adjacency_matrix(self, edges: np.ndarray, is_undirected: bool) -> None:
        '''Create the adjacency matrix from the edges'''
        max_node_index = int(np.amax(edges[:, :2])) + 1
        self.adjacency_matrix = np.zeros((max_node_index, max_node_index))
        for source, target, weight in edges:
            self.adjacency_matrix[source, target] = weight
            if is_undirected:
                self.adjacency_matrix[target, source] = weight

    def load_node_names(self, dataset_path: str) -> None:
        '''Load node labels from the given path'''
        label_file_path = f"{dataset_path}/node_labels.tsv"
        self.node_labels = pd.read_csv(label_file_path, sep="\t")

    def row_normalize_adj(self) -> None:
        '''Normalize the adjacency matrix'''
        out_degree_vector = self.adjacency_matrix.sum(axis=1)
        out_degree_vector = np.maximum(out_degree_vector, np.ones(self.num_nodes))
        inverse_out_degree = 1.0 / out_degree_vector
        inverse_out_degree_matrix = np.diag(inverse_out_degree)
        self.normalized_adjacency = inverse_out_degree_matrix.dot(self.adjacency_matrix)
        self.normalized_adjacency_transpose = self.normalized_adjacency.T
        self.out_degrees = out_degree_vector

    def power_iterate(self, teleport_prob: float = 0.15, epsilon: float = 1e-9, max_iterations: int = 100, dangling_handling: bool = False) -> Tuple[np.ndarray, List[float]]:
        '''Iterate the PageRank algorithm'''
        jump_vector = np.ones(self.num_nodes) / self.num_nodes
        previous_pagerank = jump_vector # alternatives: np.zeros(self.num_nodes) or np.random.rand(self.num_nodes)/np.random.rand(self.num_nodes).sum()
        residuals = []

        for _ in range(max_iterations):
            if dangling_handling:
                pagerank = (1 - teleport_prob) * self.normalized_adjacency_transpose.dot(previous_pagerank)
                d = np.linalg.norm(previous_pagerank, 1) - np.linalg.norm(pagerank, 1)
                pagerank = pagerank + d * jump_vector
            else:
                pagerank = (1 - teleport_prob) * self.normalized_adjacency_transpose.dot(previous_pagerank) + (teleport_prob * jump_vector)
            residual = np.linalg.norm(pagerank - previous_pagerank, 1)
            residuals.append(residual)
            previous_pagerank = pagerank

            if residual < epsilon:
                break

        return pagerank, residuals

    def closed_form(self, teleport_prob: float = 0.15) -> np.ndarray:
        '''Compute the exact PageRank score'''
        jump_vector = np.ones(self.num_nodes) / self.num_nodes
        identity_matrix = np.eye(self.num_nodes)
        H_matrix = identity_matrix - (1.0 - teleport_prob) * self.normalized_adjacency_transpose
        inverse_H_matrix = np.linalg.inv(H_matrix)
        exact_pagerank = teleport_prob * (inverse_H_matrix.dot(jump_vector))
        return exact_pagerank

    def sort_nodes_by_score(self, ranking_scores: np.ndarray, top_elements: int = -1) -> pd.DataFrame:
        '''Rank nodes by their scores'''
        sorted_node_indices = np.flipud(np.argsort(ranking_scores))
        sorted_scores = ranking_scores[sorted_node_indices]
        ranks = range(1, self.num_nodes + 1)

        top_node_labels = self.node_labels.iloc[sorted_node_indices][0:top_elements]
        top_node_labels.insert(0, "rank", ranks[0:top_elements])
        top_node_labels["score"] = sorted_scores[0:top_elements]
        top_node_labels.reset_index(drop=True, inplace=True)
        return top_node_labels

    def compute_sparsity(self) -> float:
        '''Compute the sparsity of the network'''
        sparsity = 1.0 - self.num_edges / (self.num_nodes * self.num_nodes)
        return sparsity