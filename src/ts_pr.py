import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spdiags
from typing import List, Tuple

class TopicSpecificPageRank:
    def __init__(self):
        self.adjacency_matrix: csr_matrix = None
        self.num_nodes: int = 0
        self.num_edges: int = 0
        self.node_labels: pd.DataFrame = None
        self.node_topics: pd.DataFrame = None
        self.topics: pd.DataFrame = None
        self.normalized_adjacency: csr_matrix = None
        self.normalized_adjacency_transpose: csr_matrix = None
        self.out_degrees: np.ndarray = None

    def load_network(self, dataset_path: str, is_undirected: bool = False) -> None:
        '''Load the graph dataset from the given directory (dataset_path)'''
        edges = self._load_edges(dataset_path)
        self._create_adjacency_matrix(edges, is_undirected)
        self.num_nodes = self.adjacency_matrix.shape[0]
        self.num_edges = self.adjacency_matrix.nnz

    def _load_edges(self, dataset_path: str) -> np.ndarray:
        '''Load edges from the dataset path'''
        edge_file_path = f"{dataset_path}/edges.tsv"
        return np.loadtxt(edge_file_path, dtype=int)

    def _create_adjacency_matrix(self, edges: np.ndarray, is_undirected: bool) -> None:
        '''Create the adjacency matrix from the edges'''
        num_nodes = int(np.amax(edges[:, :2])) + 1
        rows, cols, weights = edges[:, 0], edges[:, 1], edges[:, 2]
        self.adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
        if is_undirected:
            self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T

    def load_node_names(self, dataset_path: str) -> None:
        '''Load node labels from the given path'''
        label_file_path = f"{dataset_path}/node_labels.tsv"
        self.node_labels = pd.read_csv(label_file_path, sep="\t")
        
    def load_node_topics(self, dataset_path: str) -> None:
        '''Load the nodes topic labels from the given datset directory'''
        topic_path = "{}/node_topics.tsv".format(dataset_path)
        self.node_topics = pd.read_csv(topic_path, sep="\t")
        
    def load_topics(self, dataset_path: str) -> None:
        '''Load the topic data from the given datset directory'''
        topic_path = "{}/topics.tsv".format(dataset_path)
        self.topics = pd.read_csv(topic_path, sep="\t")
        
    def topic_selection(self, topic: int, verbose: bool = True) -> List[int]:
        '''Select nodes belonging to a specific topic'''
        labels = self.node_topics.iloc[:, 1]
        topic_name = self.topics.loc[topic].iloc[1]
        if verbose:
            print(f"The topic selected is: {topic} {topic_name}")
        topic_list = np.where(labels == topic)[0].tolist()
        return topic_list

    def row_normalize_adj(self) -> None:
        '''Normalize the adjacency matrix'''
        out_degree_vector = self.adjacency_matrix.sum(axis=1)
        out_degree_vector = np.maximum(np.asarray(out_degree_vector).flatten(), np.ones(self.num_nodes))
        inverse_out_degree = 1.0 / out_degree_vector
        inverse_out_degree_matrix = spdiags(inverse_out_degree, 0, self.num_nodes, self.num_nodes)
        self.normalized_adjacency = inverse_out_degree_matrix.dot(self.adjacency_matrix)
        self.normalized_adjacency_transpose = self.normalized_adjacency.T
        self.out_degrees = out_degree_vector

    def power_iterate(self, weights: List[int], teleport_prob: float = 0.15, epsilon: float = 1e-9, max_iterations: int = 100, dangling_handling: bool = False) -> Tuple[np.ndarray, List[float]]:
        '''Iterate the Personalized PageRank algorithm'''
        jump_vector = np.zeros(self.num_nodes)
        jump_vector[weights] = 1.0 / len(weights)
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

    def closed_form(self, weights: List[int], teleport_prob: float = 0.15) -> np.ndarray:
        '''Compute the exact Personalized PageRank score vector from the closed form'''
        jump_vector = np.zeros(self.num_nodes)
        jump_vector[weights] = 1.0 / len(weights)

        H_matrix = np.eye(self.num_nodes) - (1.0 - teleport_prob) * self.normalized_adjacency_transpose
        exact_scores = teleport_prob * np.linalg.inv(H_matrix).dot(jump_vector)

        return exact_scores.flatten()

    def sort_nodes_by_score(self, ranking_scores: np.ndarray, top_elements: int = -1) -> pd.DataFrame:
        '''Rank nodes by their scores'''
        sorted_indices = np.argsort(ranking_scores)[::-1]
        sorted_scores = ranking_scores[sorted_indices]
        ranks = range(1, len(sorted_indices) + 1)

        top_node_labels = self.node_labels.iloc[sorted_indices][:top_elements]
        top_node_labels.insert(0, "rank", ranks[:top_elements])
        top_node_labels["score"] = sorted_scores[:top_elements]
        top_node_labels.reset_index(drop=True, inplace=True)
        return top_node_labels

    def compute_sparsity(self) -> float:
        '''Compute the sparsity of the network'''
        sparsity = 1.0 - self.num_edges / (self.num_nodes ** 2)
        return sparsity

    def sum_pagerank_scores(self, pagerank_scores: np.ndarray) -> str:
        '''Return the sum of the PageRank score vector as a string'''
        return f"The sum of the PageRank score vector is: {np.sum(pagerank_scores):.2f}"