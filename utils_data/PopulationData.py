import ast
import os
from typing import Self, Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import Tensor
from torch_geometric.data import Data


class PopulationData:
    """
    A class to handle population data and perform various encoding operations.

    Attributes:
        data_df (pd.DataFrame): DataFrame to store population data.
        graph_nx (nx.Graph): NetworkX graph to represent connections.
    """

    def __init__(self):
        """
        Initializes the PopulationData class with empty data attributes.
        """
        self.data_df = None
        self.graph_nx = None

    def load_raw(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Self:
        """
        Loads raw data from a CSV file and drops the 'id' column.

        Args:
            path (str | bytes | os.PathLike[str] | os.PathLike[bytes]): Path to the CSV file.

        Returns:
            Self: The instance of the class with loaded data.
        """
        self.data_df = pd.read_csv(path, index_col='ID')
        return self

    def load_processed(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Self:
        """
        Loads processed data from a CSV file.

        Args:
            path (str | bytes | os.PathLike[str] | os.PathLike[bytes]): Path to the CSV file.

        Returns:
            Self: The instance of the class with loaded data.
        """
        self.data_df = pd.read_csv(path, index_col='ID')
        return self

    def save_processed(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        """
        Saves the processed data to a CSV file.

        Args:
            path (str | bytes | os.PathLike[str] | os.PathLike[bytes]): Path to save the CSV file.
        """
        self.data_df.to_csv(path)

    def save_predicted_probabilities(self,
                                     path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                                     probabilities: list[float] | pd.Series | np.ndarray[np.float_] | Tensor) -> None:
        """
        Saves the predicted probabilities to a CSV file.

        Args:
            path (str | bytes | os.PathLike[str] | os.PathLike[bytes]): Path to save the CSV file.
            probabilities (list[float] | pd.Series | np.ndarray[np.float_] | Tensor): Predicted probabilities.
        """
        if not self.is_test():
            raise ValueError("The data is not a test set.")
        assert len(self.data_df) == len(probabilities), "The number of probabilities must match the number of samples."
        out_probabilities = pd.Series(probabilities, name='Infected')
        out_df = pd.DataFrame(out_probabilities, index=self.data_df['id'])
        out_df.to_csv(path, index=True)

    def is_test(self) -> bool:
        """
        Checks if the data is a test set.

        Returns:
            bool: True if the data is a test set, False otherwise.
        """
        return 'Infected' not in self.data_df.columns

    def encode_normalized_age(self) -> Self:
        """
        Normalizes the 'Age' column using MinMaxScaler and adds it as 'Normalized_Age'.

        Returns:
            Self: The instance of the class with normalized age.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Age'] = scaler.fit_transform(self.data_df[['Age']])
        return self

    def encode_normalized_behavior(self) -> Self:
        """
        Normalizes the 'Behaviour' column using MinMaxScaler and adds it as 'Normalized_Behaviour'.

        Returns:
            Self: The instance of the class with normalized behavior.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Behaviour'] = scaler.fit_transform(self.data_df[['Behaviour']])
        return self

    def encode_normalized_constitution(self) -> Self:
        """
        Normalizes the 'Constitution' column using MinMaxScaler and adds it as 'Normalized_Constitution'.

        Returns:
            Self: The instance of the class with normalized constitution.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Constitution'] = scaler.fit_transform(self.data_df[['Constitution']])
        return self

    def encode_standardized_age(self) -> Self:
        """
        Standardizes the 'Age' column using StandardScaler and adds it as 'Standardized_Age'.

        Returns:
            Self: The instance of the class with standardized age.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Age'] = scaler.fit_transform(self.data_df[['Age']])
        return self

    def encode_standardized_constitution(self) -> Self:
        """
        Standardizes the 'Constitution' column using StandardScaler and adds it as 'Standardized_Constitution'.

        Returns:
            Self: The instance of the class with standardized constitution.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Constitution'] = scaler.fit_transform(self.data_df[['Constitution']])
        return self

    def encode_standardized_behavior(self) -> Self:
        """
        Standardizes the 'Behaviour' column using StandardScaler and adds it as 'Standardized_Behaviour'.

        Returns:
            Self: The instance of the class with standardized behavior.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Behaviour'] = scaler.fit_transform(self.data_df[['Behaviour']])
        return self

    def encode_connection_lists(self) -> Self:
        """
        Converts the 'Connections' column from string representation of lists to actual lists.

        Returns:
            Self: The instance of the class with encoded connection lists.
        """
        self.data_df['Connections'] = self.data_df['Connections'].apply(ast.literal_eval)
        return self

    def encode_connection_int(self) -> Self:
        """
        Maps node identifiers to integers and updates the 'Connections' column accordingly.

        Returns:
            Self: The instance of the class with encoded connection integers.
        """
        all_nodes = set(self.data_df.index).union(*self.data_df['Connections'].apply(ast.literal_eval))
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        self.data_df.index = self.data_df.index.map(node_mapping)
        self.data_df['Connections'] = self.data_df['Connections'].apply(lambda conn_list: [node_mapping[conn] for conn in ast.literal_eval(conn_list)])
        return self

    def encode_graph_nx(self) -> Self:
        """
        Creates a NetworkX graph from the 'Connections' column.

        Returns:
            Self: The instance of the class with encoded NetworkX graph.
        """
        self.graph_nx = nx.Graph()
        for idx, row in self.data_df.iterrows():
            self.graph_nx.add_node(idx)
            for connection in row['Connections']:
                self.graph_nx.add_edge(idx, connection)
        return self

    def encode_node2vec(self,
                        dimensions: int = 64,
                        walk_length: int = 64,
                        num_walks: int = 16,
                        p: int = 1,
                        q: int = 1,
                        workers: int = 4,
                        window: int = 8,
                        min_count: int = 1,
                        batch_words: int = 16) -> Self:
        """
        Encodes nodes using Node2Vec algorithm and adds the embeddings to the DataFrame.

        Args:
            dimensions (int): Number of dimensions for the embeddings.
            walk_length (int): Length of each walk.
            num_walks (int): Number of walks per node.
            p (int): Return hyperparameter.
            q (int): Inout hyperparameter.
            workers (int): Number of workers for parallel processing.
            window (int): Window size for Word2Vec.
            min_count (int): Minimum count for Word2Vec.
            batch_words (int): Batch size for Word2Vec.

        Returns:
            Self: The instance of the class with Node2Vec embeddings.
        """
        node2vec = Node2Vec(self.graph_nx,
                            dimensions=dimensions,
                            walk_length=walk_length,
                            num_walks=num_walks,
                            p=p,
                            q=q,
                            workers=workers)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        embeddings_df = pd.DataFrame(embeddings).T
        embeddings_df.columns = [f'Node2Vec_{i}' for i in range(dimensions)]
        self.data_df = self.data_df.join(embeddings_df)
        return self

    def encode_degree(self) -> Self:
        """
        Calculates the degree of each node and adds it as 'Degree' to the DataFrame.

        Returns:
            Self: The instance of the class with node degrees.
        """
        degree_dict = dict(self.graph_nx.degree())
        degree_series = pd.Series(degree_dict)
        self.data_df['Degree'] = degree_series
        return self

    def encode_degree_centrality(self) -> Self:
        """
        Calculates the degree centrality of each node and adds it as 'Degree_Centrality' to the DataFrame.

        Returns:
            Self: The instance of the class with degree centrality.
        """
        degree_centrality_dict = nx.degree_centrality(self.graph_nx)
        degree_centrality_series = pd.Series(degree_centrality_dict)
        self.data_df['Degree_Centrality'] = degree_centrality_series
        return self

    def encode_clustering_coefficient(self) -> Self:
        """
        Calculates the clustering coefficient of each node and adds it as 'Clustering_Coefficient' to the DataFrame.

        Returns:
            Self: The instance of the class with clustering coefficients.
        """
        clustering_coefficient_dict = nx.clustering(self.graph_nx)
        clustering_coefficient_series = pd.Series(clustering_coefficient_dict)
        self.data_df['Clustering_Coefficient'] = clustering_coefficient_series
        return self

    def encode_normalized_degree(self) -> Self:
        """
        Normalizes the 'Degree' column using MinMaxScaler and adds it as 'Normalized_Degree'.

        Returns:
            Self: The instance of the class with normalized degree.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Degree'] = scaler.fit_transform(self.data_df[['Degree']])
        return self

    def encode_normalized_degree_centrality(self) -> Self:
        """
        Normalizes the 'Degree_Centrality' column using MinMaxScaler and adds it as 'Normalized_Degree_Centrality'.

        Returns:
            Self: The instance of the class with normalized degree centrality.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Degree_Centrality']])
        return self

    def encode_normalized_clustering_coefficient(self) -> Self:
        """
        Normalizes the 'Clustering_Coefficient' column using MinMaxScaler and adds it as 'Normalized_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with normalized clustering coefficient.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Clustering_Coefficient']])
        return self

    def encode_standardized_degree(self) -> Self:
        """
        Standardizes the 'Degree' column using StandardScaler and adds it as 'Standardized_Degree'.

        Returns:
            Self: The instance of the class with standardized degree.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Degree'] = scaler.fit_transform(self.data_df[['Degree']])
        return self

    def encode_standardized_degree_centrality(self) -> Self:
        """
        Standardizes the 'Degree_Centrality' column using StandardScaler and adds it as 'Standardized_Degree_Centrality'.

        Returns:
            Self: The instance of the class with standardized degree centrality.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Degree_Centrality']])
        return self

    def encode_standardized_clustering_coefficient(self) -> Self:
        """
        Standardizes the 'Clustering_Coefficient' column using StandardScaler and adds it as 'Standardized_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with standardized clustering coefficient.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Clustering_Coefficient']])
        return self

    def encode_connected_index_patient(self) -> Self:
        """
        Maps each node to its connected index patient and adds it as 'Connected_Index_Patient'.

        Returns:
            Self: The instance of the class with connected index patients.
        """
        index_patients = self.data_df[self.data_df['Index_Patient'] == 1]
        index_patients_dict = dict(zip(index_patients['Population'], index_patients.index))
        self.data_df['Connected_Index_Patient'] = self.data_df.apply(
            lambda row: index_patients_dict[row['Population']], axis=1
        )
        return self

    def encode_distance_to_index_patient(self) -> Self:
        """
        Calculates the shortest path distance to the nearest index patient and adds it as 'Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with distances to index patients.
        """
        index_patients = self.data_df[self.data_df['Index_Patient'] == 1].index.to_list()
        shortest_paths_all = nx.multi_source_dijkstra_path_length(self.graph_nx, index_patients)
        shortest_paths = {node: float('inf') for node in self.data_df.index}
        for node, length in shortest_paths_all.items():
            shortest_paths[node] = length
        self.data_df['Distance_to_Index_Patient'] = pd.Series(shortest_paths)
        return self

    def encode_normalized_distance_to_index_patient(self) -> Self:
        """
        Normalizes the 'Distance_to_Index_Patient' column using MinMaxScaler and adds it as 'Normalized_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with normalized distances to index patients.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Distance_to_Index_Patient']])
        return self

    def encode_standardized_distance_to_index_patient(self) -> Self:
        """
        Standardizes the 'Distance_to_Index_Patient' column using StandardScaler and adds it as 'Standardized_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with standardized distances to index patients.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Distance_to_Index_Patient']])
        return self

    def encode_sum_neighbor_age(self) -> Self:
        """
        Calculates the sum of ages of neighboring nodes and adds it as 'Sum_Neighbor_Age'.

        Returns:
            Self: The instance of the class with sum of neighbor ages.
        """
        ages = self.data_df['Age'].to_dict()
        self.data_df['Sum_Neighbor_Age'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([ages[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_constitution(self) -> Self:
        """
        Calculates the sum of constitutions of neighboring nodes and adds it as 'Sum_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with sum of neighbor constitutions.
        """
        constitutions = self.data_df['Constitution'].to_dict()
        self.data_df['Sum_Neighbor_Constitution'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([constitutions[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_behavior(self) -> Self:
        """
        Calculates the sum of behaviors of neighboring nodes and adds it as 'Sum_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with sum of neighbor behaviors.
        """
        behaviors = self.data_df['Behaviour'].to_dict()
        self.data_df['Sum_Neighbor_Behaviour'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([behaviors[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_degree(self) -> Self:
        """
        Calculates the sum of degrees of neighboring nodes and adds it as 'Sum_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with sum of neighbor degrees.
        """
        degree = self.data_df['Degree'].to_dict()
        self.data_df['Sum_Neighbor_Degree'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([degree[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_degree_centrality(self) -> Self:
        """
        Calculates the sum of degree centralities of neighboring nodes and adds it as 'Sum_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with sum of neighbor degree centralities.
        """
        degree_centrality = self.data_df['Degree_Centrality'].to_dict()
        self.data_df['Sum_Neighbor_Degree_Centrality'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([degree_centrality[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_clustering_coefficient(self) -> Self:
        """
        Calculates the sum of clustering coefficients of neighboring nodes and adds it as 'Sum_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with sum of neighbor clustering coefficients.
        """
        clustering_coefficient = self.data_df['Clustering_Coefficient'].to_dict()
        self.data_df['Sum_Neighbor_Clustering_Coefficient'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([clustering_coefficient[connection] for connection in connections])
        )
        return self

    def encode_normalized_sum_neighbor_age(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Age' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Age'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor ages.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Age']])
        return self

    def encode_normalized_sum_neighbor_constitution(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Constitution' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor constitutions.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Constitution']])
        return self

    def encode_normalized_sum_neighbor_behavior(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Behaviour' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor behaviors.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Behaviour']])
        return self

    def encode_normalized_sum_neighbor_degree(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Degree' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor degrees.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree']])
        return self

    def encode_normalized_sum_neighbor_degree_centrality(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Degree_Centrality' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor degree centralities.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree_Centrality']])
        return self

    def encode_normalized_sum_neighbor_clustering_coefficient(self) -> Self:
        """
        Normalizes the 'Sum_Neighbor_Clustering_Coefficient' column using MinMaxScaler and adds it as 'Normalized_Sum_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with normalized sum of neighbor clustering coefficients.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Clustering_Coefficient']])
        return self

    def encode_standardized_sum_neighbor_age(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Age' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Age'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor ages.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Age']])
        return self

    def encode_standardized_sum_neighbor_constitution(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Constitution' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor constitutions.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Constitution']])
        return self

    def encode_standardized_sum_neighbor_behavior(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Behaviour' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor behaviors.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Behaviour']])
        return self

    def encode_standardized_sum_neighbor_degree(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Degree' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor degrees.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree']])
        return self

    def encode_standardized_sum_neighbor_degree_centrality(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Degree_Centrality' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor degree centralities.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree_Centrality']])
        return self

    def encode_standardized_sum_neighbor_clustering_coefficient(self) -> Self:
        """
        Standardizes the 'Sum_Neighbor_Clustering_Coefficient' column using StandardScaler and adds it as 'Standardized_Sum_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with standardized sum of neighbor clustering coefficients.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Clustering_Coefficient']])
        return self

    def encode_mean_neighbor_age(self) -> Self:
        """
        Calculates the mean age of neighboring nodes and adds it as 'Mean_Neighbor_Age'.

        Returns:
            Self: The instance of the class with mean neighbor ages.
        """
        ages = self.data_df['Age'].to_dict()
        self.data_df['Mean_Neighbor_Age'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([ages[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_constitution(self) -> Self:
        """
        Calculates the mean constitution of neighboring nodes and adds it as 'Mean_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with mean neighbor constitutions.
        """
        constitutions = self.data_df['Constitution'].to_dict()
        self.data_df['Mean_Neighbor_Constitution'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([constitutions[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_behavior(self) -> Self:
        """
        Calculates the mean behavior of neighboring nodes and adds it as 'Mean_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with mean neighbor behaviors.
        """
        behaviors = self.data_df['Behaviour'].to_dict()
        self.data_df['Mean_Neighbor_Behaviour'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([behaviors[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_degree(self) -> Self:
        """
        Calculates the mean degree of neighboring nodes and adds it as 'Mean_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with mean neighbor degrees.
        """
        degree = self.data_df['Degree'].to_dict()
        self.data_df['Mean_Neighbor_Degree'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([degree[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_degree_centrality(self) -> Self:
        """
        Calculates the mean degree centrality of neighboring nodes and adds it as 'Mean_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with mean neighbor degree centralities.
        """
        degree_centrality = self.data_df['Degree_Centrality'].to_dict()
        self.data_df['Mean_Neighbor_Degree_Centrality'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([degree_centrality[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_clustering_coefficient(self) -> Self:
        """
        Calculates the mean clustering coefficient of neighboring nodes and adds it as 'Mean_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with mean neighbor clustering coefficients.
        """
        clustering_coefficient = self.data_df['Clustering_Coefficient'].to_dict()
        self.data_df['Mean_Neighbor_Clustering_Coefficient'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([clustering_coefficient[connection] for connection in connections])
        )
        return self

    def encode_normalized_mean_neighbor_age(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Age' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Age'.

        Returns:
            Self: The instance of the class with normalized mean neighbor ages.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Age']])
        return self

    def encode_normalized_mean_neighbor_constitution(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Constitution' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with normalized mean neighbor constitutions.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Constitution']])
        return self

    def encode_normalized_mean_neighbor_behavior(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Behaviour' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with normalized mean neighbor behaviors.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Behaviour']])
        return self

    def encode_normalized_mean_neighbor_degree(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Degree' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with normalized mean neighbor degrees.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree']])
        return self

    def encode_normalized_mean_neighbor_degree_centrality(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Degree_Centrality' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with normalized mean neighbor degree centralities.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree_Centrality']])
        return self

    def encode_normalized_mean_neighbor_clustering_coefficient(self) -> Self:
        """
        Normalizes the 'Mean_Neighbor_Clustering_Coefficient' column using MinMaxScaler and adds it as 'Normalized_Mean_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with normalized mean neighbor clustering coefficients.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Clustering_Coefficient']])
        return self

    def encode_standardized_mean_neighbor_age(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Age' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Age'.

        Returns:
            Self: The instance of the class with standardized mean neighbor ages.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Age']])
        return self

    def encode_standardized_mean_neighbor_constitution(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Constitution' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Constitution'.

        Returns:
            Self: The instance of the class with standardized mean neighbor constitutions.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Constitution']])
        return self

    def encode_standardized_mean_neighbor_behavior(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Behaviour' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Behaviour'.

        Returns:
            Self: The instance of the class with standardized mean neighbor behaviors.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Behaviour']])
        return self

    def encode_standardized_mean_neighbor_degree(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Degree' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Degree'.

        Returns:
            Self: The instance of the class with standardized mean neighbor degrees.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree']])
        return self

    def encode_standardized_mean_neighbor_degree_centrality(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Degree_Centrality' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Degree_Centrality'.

        Returns:
            Self: The instance of the class with standardized mean neighbor degree centralities.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree_Centrality']])
        return self

    def encode_standardized_mean_neighbor_clustering_coefficient(self) -> Self:
        """
        Standardizes the 'Mean_Neighbor_Clustering_Coefficient' column using StandardScaler and adds it as 'Standardized_Mean_Neighbor_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with standardized mean neighbor clustering coefficients.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Clustering_Coefficient']])
        return self

    def encode_sum_population_age(self) -> Self:
        """
        Calculates the sum of ages for each population and adds it as 'Sum_Population_Age'.

        Returns:
            Self: The instance of the class with sum of population ages.
        """
        sum_population_age = self.data_df.groupby('Population')['Age'].sum()
        self.data_df['Sum_Population_Age'] = self.data_df['Population'].map(sum_population_age)
        return self

    def encode_sum_population_constitution(self) -> Self:
        """
        Calculates the sum of constitutions for each population and adds it as 'Sum_Population_Constitution'.

        Returns:
            Self: The instance of the class with sum of population constitutions.
        """
        sum_population_constitution = self.data_df.groupby('Population')['Constitution'].sum()
        self.data_df['Sum_Population_Constitution'] = self.data_df['Population'].map(sum_population_constitution)
        return self

    def encode_sum_population_behavior(self) -> Self:
        """
        Calculates the sum of behaviors for each population and adds it as 'Sum_Population_Behaviour'.

        Returns:
            Self: The instance of the class with sum of population behaviors.
        """
        sum_population_behavior = self.data_df.groupby('Population')['Behaviour'].sum()
        self.data_df['Sum_Population_Behaviour'] = self.data_df['Population'].map(sum_population_behavior)
        return self

    def encode_sum_population_degree(self) -> Self:
        """
        Calculates the sum of degrees for each population and adds it as 'Sum_Population_Degree'.

        Returns:
            Self: The instance of the class with sum of population degrees.
        """
        sum_population_degree = self.data_df.groupby('Population')['Degree'].sum()
        self.data_df['Sum_Population_Degree'] = self.data_df['Population'].map(sum_population_degree)
        return self

    def encode_sum_population_degree_centrality(self) -> Self:
        """
        Calculates the sum of degree centralities for each population and adds it as 'Sum_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with sum of population degree centralities.
        """
        sum_population_degree_centrality = self.data_df.groupby('Population')['Degree_Centrality'].sum()
        self.data_df['Sum_Population_Degree_Centrality'] = self.data_df['Population'].map(sum_population_degree_centrality)
        return self

    def encode_sum_population_clustering_coefficient(self) -> Self:
        """
        Calculates the sum of clustering coefficients for each population and adds it as 'Sum_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with sum of population clustering coefficients.
        """
        sum_population_clustering_coefficient = self.data_df.groupby('Population')['Clustering_Coefficient'].sum()
        self.data_df['Sum_Population_Clustering_Coefficient'] = self.data_df['Population'].map(sum_population_clustering_coefficient)
        return self

    def encode_normalized_sum_population_age(self) -> Self:
        """
        Normalizes the 'Sum_Population_Age' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Age'.

        Returns:
            Self: The instance of the class with normalized sum of population ages.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Age'] = scaler.fit_transform(self.data_df[['Sum_Population_Age']])
        return self

    def encode_normalized_sum_population_constitution(self) -> Self:
        """
        Normalizes the 'Sum_Population_Constitution' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Constitution'.

        Returns:
            Self: The instance of the class with normalized sum of population constitutions.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Population_Constitution']])
        return self

    def encode_normalized_sum_population_behavior(self) -> Self:
        """
        Normalizes the 'Sum_Population_Behaviour' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Behaviour'.

        Returns:
            Self: The instance of the class with normalized sum of population behaviors.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Population_Behaviour']])
        return self

    def encode_normalized_sum_population_degree(self) -> Self:
        """
        Normalizes the 'Sum_Population_Degree' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Degree'.

        Returns:
            Self: The instance of the class with normalized sum of population degrees.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Degree'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree']])
        return self

    def encode_normalized_sum_population_degree_centrality(self) -> Self:
        """
        Normalizes the 'Sum_Population_Degree_Centrality' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with normalized sum of population degree centralities.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree_Centrality']])
        return self

    def encode_normalized_sum_population_clustering_coefficient(self) -> Self:
        """
        Normalizes the 'Sum_Population_Clustering_Coefficient' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with normalized sum of population clustering coefficients.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Population_Clustering_Coefficient']])
        return self

    def encode_standardized_sum_population_age(self) -> Self:
        """
        Standardizes the 'Sum_Population_Age' column using StandardScaler and adds it as 'Standardized_Sum_Population_Age'.

        Returns:
            Self: The instance of the class with standardized sum of population ages.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Age'] = scaler.fit_transform(self.data_df[['Sum_Population_Age']])
        return self

    def encode_standardized_sum_population_constitution(self) -> Self:
        """
        Standardizes the 'Sum_Population_Constitution' column using StandardScaler and adds it as 'Standardized_Sum_Population_Constitution'.

        Returns:
            Self: The instance of the class with standardized sum of population constitutions.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Population_Constitution']])
        return self

    def encode_standardized_sum_population_behavior(self) -> Self:
        """
        Standardizes the 'Sum_Population_Behaviour' column using StandardScaler and adds it as 'Standardized_Sum_Population_Behaviour'.

        Returns:
            Self: The instance of the class with standardized sum of population behaviors.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Population_Behaviour']])
        return self

    def encode_standardized_sum_population_degree(self) -> Self:
        """
        Standardizes the 'Sum_Population_Degree' column using StandardScaler and adds it as 'Standardized_Sum_Population_Degree'.

        Returns:
            Self: The instance of the class with standardized sum of population degrees.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Degree'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree']])
        return self

    def encode_standardized_sum_population_degree_centrality(self) -> Self:
        """
        Standardizes the 'Sum_Population_Degree_Centrality' column using StandardScaler and adds it as 'Standardized_Sum_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with standardized sum of population degree centralities.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree_Centrality']])
        return self

    def encode_standardized_sum_population_clustering_coefficient(self) -> Self:
        """
        Standardizes the 'Sum_Population_Clustering_Coefficient' column using StandardScaler and adds it as 'Standardized_Sum_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with standardized sum of population clustering coefficients.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Population_Clustering_Coefficient']])
        return self

    def encode_mean_population_age(self) -> Self:
        """
        Calculates the mean age for each population and adds it as 'Mean_Population_Age'.

        Returns:
            Self: The instance of the class with mean population ages.
        """
        mean_population_age = self.data_df.groupby('Population')['Age'].mean()
        self.data_df['Mean_Population_Age'] = self.data_df['Population'].map(mean_population_age)
        return self

    def encode_mean_population_constitution(self) -> Self:
        """
        Calculates the mean constitution for each population and adds it as 'Mean_Population_Constitution'.

        Returns:
            Self: The instance of the class with mean population constitutions.
        """
        mean_population_constitution = self.data_df.groupby('Population')['Constitution'].mean()
        self.data_df['Mean_Population_Constitution'] = self.data_df['Population'].map(mean_population_constitution)
        return self

    def encode_mean_population_behavior(self) -> Self:
        """
        Calculates the mean behavior for each population and adds it as 'Mean_Population_Behaviour'.

        Returns:
            Self: The instance of the class with mean population behaviors.
        """
        mean_population_behavior = self.data_df.groupby('Population')['Behaviour'].mean()
        self.data_df['Mean_Population_Behaviour'] = self.data_df['Population'].map(mean_population_behavior)
        return self

    def encode_mean_population_degree(self) -> Self:
        """
        Calculates the mean degree for each population and adds it as 'Mean_Population_Degree'.

        Returns:
            Self: The instance of the class with mean population degrees.
        """
        mean_population_degree = self.data_df.groupby('Population')['Degree'].mean()
        self.data_df['Mean_Population_Degree'] = self.data_df['Population'].map(mean_population_degree)
        return self

    def encode_mean_population_degree_centrality(self) -> Self:
        """
        Calculates the mean degree centrality for each population and adds it as 'Mean_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with mean population degree centralities.
        """
        mean_population_degree_centrality = self.data_df.groupby('Population')['Degree_Centrality'].mean()
        self.data_df['Mean_Population_Degree_Centrality'] = self.data_df['Population'].map(mean_population_degree_centrality)
        return self

    def encode_mean_population_clustering_coefficient(self) -> Self:
        """
        Calculates the mean clustering coefficient for each population and adds it as 'Mean_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with mean population clustering coefficients.
        """
        mean_population_clustering_coefficient = self.data_df.groupby('Population')['Clustering_Coefficient'].mean()
        self.data_df['Mean_Population_Clustering_Coefficient'] = self.data_df['Population'].map(mean_population_clustering_coefficient)
        return self

    def encode_normalized_mean_population_age(self) -> Self:
        """
        Normalizes the 'Mean_Population_Age' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Age'.

        Returns:
            Self: The instance of the class with the normalized mean population age.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Age'] = scaler.fit_transform(self.data_df[['Mean_Population_Age']])
        return self

    def encode_normalized_mean_population_constitution(self) -> Self:
        """
        Normalizes the 'Mean_Population_Constitution' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Constitution'.

        Returns:
            Self: The instance of the class with the normalized mean population constitution.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Population_Constitution']])
        return self

    def encode_normalized_mean_population_behavior(self) -> Self:
        """
        Normalizes the 'Mean_Population_Behaviour' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Behaviour'.

        Returns:
            Self: The instance of the class with the normalized mean population behavior.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Population_Behaviour']])
        return self

    def encode_normalized_mean_population_degree(self) -> Self:
        """
        Normalizes the 'Mean_Population_Degree' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Degree'.

        Returns:
            Self: The instance of the class with the normalized mean population degree.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Degree'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree']])
        return self

    def encode_normalized_mean_population_degree_centrality(self) -> Self:
        """
        Normalizes the 'Mean_Population_Degree_Centrality' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with the normalized mean population degree centrality.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree_Centrality']])
        return self

    def encode_normalized_mean_population_clustering_coefficient(self) -> Self:
        """
        Normalizes the 'Mean_Population_Clustering_Coefficient' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with the normalized mean population clustering coefficient.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Population_Clustering_Coefficient']])
        return self

    def encode_standardized_mean_population_age(self) -> Self:
        """
        Standardizes the 'Mean_Population_Age' column using StandardScaler and adds it as 'Standardized_Mean_Population_Age'.

        Returns:
            Self: The instance of the class with the standardized mean population age.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Age'] = scaler.fit_transform(self.data_df[['Mean_Population_Age']])
        return self

    def encode_standardized_mean_population_constitution(self) -> Self:
        """
        Standardizes the 'Mean_Population_Constitution' column using StandardScaler and adds it as 'Standardized_Mean_Population_Constitution'.

        Returns:
            Self: The instance of the class with the standardized mean population constitution.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Population_Constitution']])
        return self

    def encode_standardized_mean_population_behavior(self) -> Self:
        """
        Standardizes the 'Mean_Population_Behaviour' column using StandardScaler and adds it as 'Standardized_Mean_Population_Behaviour'.

        Returns:
            Self: The instance of the class with the standardized mean population behavior.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Population_Behaviour']])
        return self

    def encode_standardized_mean_population_degree(self) -> Self:
        """
        Standardizes the 'Mean_Population_Degree' column using StandardScaler and adds it as 'Standardized_Mean_Population_Degree'.

        Returns:
            Self: The instance of the class with the standardized mean population degree.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Degree'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree']])
        return self

    def encode_standardized_mean_population_degree_centrality(self) -> Self:
        """
        Standardizes the 'Mean_Population_Degree_Centrality' column using StandardScaler and adds it as 'Standardized_Mean_Population_Degree_Centrality'.

        Returns:
            Self: The instance of the class with the standardized mean population degree centrality.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree_Centrality']])
        return self

    def encode_standardized_mean_population_clustering_coefficient(self) -> Self:
        """
        Standardizes the 'Mean_Population_Clustering_Coefficient' column using StandardScaler and adds it as 'Standardized_Mean_Population_Clustering_Coefficient'.

        Returns:
            Self: The instance of the class with the standardized mean population clustering coefficient.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Population_Clustering_Coefficient']])
        return self

    def encode_sum_population_distance_to_index_patient(self) -> Self:
        """
        Calculates the sum of distances to the index patient for each population and adds it as 'Sum_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the sum of distances to the index patient.
        """
        sum_population_distance_to_index_patient = self.data_df.groupby('Population')['Distance_to_Index_Patient'].sum()
        self.data_df['Sum_Population_Distance_to_Index_Patient'] = self.data_df['Population'].map(sum_population_distance_to_index_patient)
        return self

    def encode_normalized_sum_population_distance_to_index_patient(self) -> Self:
        """
        Normalizes the 'Sum_Population_Distance_to_Index_Patient' column using MinMaxScaler and adds it as 'Normalized_Sum_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the normalized sum of distances to the index patient.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Sum_Population_Distance_to_Index_Patient']])
        return self

    def encode_standardized_sum_population_distance_to_index_patient(self) -> Self:
        """
        Standardizes the 'Sum_Population_Distance_to_Index_Patient' column using StandardScaler and adds it as 'Standardized_Sum_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the standardized sum of distances to the index patient.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Sum_Population_Distance_to_Index_Patient']])
        return self

    def encode_mean_population_distance_to_index_patient(self) -> Self:
        """
        Calculates the mean distance to the index patient for each population and adds it as 'Mean_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the mean distance to the index patient.
        """
        mean_population_distance_to_index_patient = self.data_df.groupby('Population')['Distance_to_Index_Patient'].mean()
        self.data_df['Mean_Population_Distance_to_Index_Patient'] = self.data_df['Population'].map(mean_population_distance_to_index_patient)
        return self

    def encode_normalized_mean_population_distance_to_index_patient(self) -> Self:
        """
        Normalizes the 'Mean_Population_Distance_to_Index_Patient' column using MinMaxScaler and adds it as 'Normalized_Mean_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the normalized mean distance to the index patient.
        """
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Mean_Population_Distance_to_Index_Patient']])
        return self

    def encode_standardized_mean_population_distance_to_index_patient(self) -> Self:
        """
        Standardizes the 'Mean_Population_Distance_to_Index_Patient' column using StandardScaler and adds it as 'Standardized_Mean_Population_Distance_to_Index_Patient'.

        Returns:
            Self: The instance of the class with the standardized mean distance to the index patient.
        """
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Mean_Population_Distance_to_Index_Patient']])
        return self

    def encode_train_eval(self) -> Self:
        """
        Splits the data into training and testing sets and adds 'Train' and 'Eval' columns to indicate the split.

        Returns:
            Self: The instance of the class with the training and testing sets.
        """
        if self.is_test():
            return self
        indices = self.data_df.index
        train_idx, test_idx = train_test_split(indices, test_size=0.2)
        self.data_df['Train'] = self.data_df.index.isin(train_idx)
        self.data_df['Eval'] = self.data_df.index.isin(test_idx)
        return self

    def get_id_dataframe(self,
                         train: Literal['Train', 'Eval'] = None,
                         population: str = None) -> pd.Series:
        """
        Retrieves the id data as a pandas Series.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            pd.Series: The id data as a pandas Series.
        """
        out = self.data_df
        if population is not None:
            out = out[out["Population"] == population]
        if not self.is_test() and train is not None:
            out = out[out[train]]
        out_ids = out["id"]
        return out_ids

    def get_id_numpy(self,
                     train: Literal['Train', 'Eval'] = None,
                     population: str = None) -> np.ndarray:
        """
        Retrieves the id data as a numpy array.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            np.ndarray: The id data as a numpy array.
        """
        ids_df = self.get_id_dataframe(train=train, population=population)
        ids_np = ids_df.to_numpy()
        return ids_np

    def get_id_tensor(self,
                      train: Literal['Train', 'Eval'] = None,
                      population: str = None) -> Tensor:
        """
        Retrieves the id data as a PyTorch tensor.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            Tensor: The id data as a PyTorch tensor.
        """
        ids_np = self.get_id_numpy(train=train, population=population)
        ids_ts = torch.tensor(ids_np, dtype=torch.float32)
        return ids_ts

    def get_feature_dataframe(self,
                              features: list[str] = None,
                              train: Literal['Train', 'Eval'] = None,
                              population: str = None) -> pd.DataFrame:
        """
        Retrieves the features data as a pandas DataFrame.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            pd.DataFrame: The features data as a pandas DataFrame.
        """
        out = self.data_df
        if population is not None:
            out = out[out["Population"] == population]
        if not self.is_test() and train is not None:
            out = out[out[train]]
        out = out.drop(columns=["id", "Population", "Connections", "Train", "Eval", "Infected"], axis=1, errors="ignore")
        out_features = out
        if features is not None:
            out_columns = [feature for feature in features if feature in out_features.columns]
            out_features = out_features[out_columns]
        return out_features

    def get_feature_numpy(self,
                          features: list[str] = None,
                          train: Literal['Train', 'Eval'] = None,
                          population: str = None) -> np.ndarray:
        """
        Retrieves the features data as a numpy array.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            np.ndarray: The features data as a numpy array.
        """
        features_df = self.get_feature_dataframe(features=features, train=train, population=population)
        features_np = features_df.to_numpy()
        return features_np

    def get_feature_tensor(self,
                           features: list[str] = None,
                           train: Literal['Train', 'Eval'] = None,
                           population: str = None) -> Tensor:
        """
        Retrieves the features data as a PyTorch tensor.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            Tensor: The features data as a PyTorch tensor.
        """
        features_np = self.get_feature_numpy(features=features, train=train, population=population)
        features_ts = torch.tensor(features_np, dtype=torch.float32)
        return features_ts

    def get_label_dataframe(self,
                            train: Literal['Train', 'Eval'] = None,
                            population: str = None) -> pd.Series:
        """
        Retrieves the labels data as a pandas Series.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            pd.Series: The labels data as a pandas Series.

        Raises:
            ValueError: If the data is test data.
        """
        if self.is_test():
            raise ValueError("Test data does not have labels.")
        out = self.data_df
        if population is not None:
            out = out[out["Population"] == population]
        if train is not None:
            out = out[out[train]]
        out_labels = out["Infected"]
        return out_labels

    def get_label_numpy(self,
                        train: Literal['Train', 'Eval'] = None,
                        population: str = None) -> np.ndarray:
        """
        Retrieves the labels data as a numpy array.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            np.ndarray: The labels data as a numpy array.
        """
        labels_df = self.get_label_dataframe(train=train, population=population)
        labels_np = labels_df.to_numpy()
        return labels_np

    def get_label_tensor(self,
                         train: Literal['Train', 'Eval'] = None,
                         population: str = None) -> Tensor:
        """
        Retrieves the labels data as a PyTorch tensor.

        Args:
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            Tensor: The labels data as a PyTorch tensor.
        """
        labels_np = self.get_label_numpy(train=train, population=population)
        labels_ts = torch.tensor(labels_np, dtype=torch.float32)
        return labels_ts

    def get_feature_label_dataframes(self,
                                     features: list[str] = None,
                                     train: Literal['Train', 'Eval'] = None,
                                     population: str = None) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Retrieves the data as pandas DataFrames.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (str, optional): Population to filter by. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.Series]: The features and labels as DataFrames.
        """
        out_features = self.get_feature_dataframe(features=features, train=train, population=population)
        out_labels = self.get_label_dataframe(train=train, population=population) if not self.is_test() else None
        return out_features, out_labels

    def get_feature_label_numpy(self,
                                features: list[str] = None,
                                train: Literal['Train', 'Eval'] = None,
                                population: str = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Retrieves the data as numpy arrays.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (str, optional): Population to filter by. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: The features and labels as numpy arrays.
        """
        features_df, labels_df = self.get_feature_label_dataframes(features=features, train=train, population=population)
        features_np = features_df.to_numpy()
        labels_np = labels_df.to_numpy() if labels_df is not None else None
        return features_np, labels_np

    def get_feature_label_tensors(self,
                                  features: list[str] = None,
                                  train: Literal['Train', 'Eval'] = None,
                                  population: str = None) -> tuple[Tensor, Tensor | None]:
        """
        Retrieves the data as PyTorch tensors.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: The features and labels as PyTorch tensors.
        """
        features_np, labels_np = self.get_feature_label_numpy(features=features, train=train, population=population)
        features_ts = torch.tensor(features_np, dtype=torch.float32)
        labels_ts = torch.tensor(labels_np, dtype=torch.float32) if labels_np is not None else None
        return features_ts, labels_ts

    def get_id_feature_dataframes(self,
                                  features: list[str] = None,
                                  train: Literal['Train', 'Eval'] = None,
                                  population: str = None) -> tuple[pd.Series, pd.DataFrame]:
        """
        Retrieves the data as pandas DataFrames.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (str, optional): Population to filter by. Defaults to None.

        Returns:
            tuple[pd.Series, pd.DataFrame]: The ids and features as DataFrames.
        """
        out_ids = self.get_id_dataframe(train=train, population=population)
        out_features = self.get_feature_dataframe(features=features, train=train, population=population)
        return out_ids, out_features

    def get_id_feature_numpy(self,
                             features: list[str] = None,
                             train: Literal['Train', 'Eval'] = None,
                             population: str = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the data as numpy arrays.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
            population (str, optional): Population to filter by. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: The ids and features as numpy arrays.
        """
        ids_df, features_df = self.get_id_feature_dataframes(features=features, train=train, population=population)
        ids_np = ids_df.to_numpy()
        features_np = features_df.to_numpy()
        return ids_np, features_np

    def get_id_feature_tensors(self,
                               features: list[str] = None,
                               train: Literal['Train', 'Eval'] = None,
                               population: str = None) -> tuple[Tensor, Tensor]:
        """
        Retrieves the data as PyTorch tensors.

        Args:
              features (list[str], optional): List of feature columns to include. Defaults to None.
              train (Literal['Train', 'Eval'], optional): Whether to retrieve training or testing data. Defaults to None.
              population (int, optional): Population to filter by. Defaults to None.

        Returns:
              tuple[Tensor, Tensor]: The ids and features as PyTorch tensors.
        """
        ids_np, features_np = self.get_id_feature_numpy(features=features, train=train, population=population)
        ids_ts = torch.tensor(ids_np, dtype=torch.float32)
        features_ts = torch.tensor(features_np, dtype=torch.float32)
        return ids_ts, features_ts

    def get_graph_nx(self,
                     features: list[str] = None,
                     population: str = None) -> nx.Graph:
        """
        Retrieves the data as a NetworkX graph.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.
            population (int, optional): Population to filter by. Defaults to None.

        Returns:
            nx.Graph: The data as a NetworkX graph.
        """
        out = self.graph_nx.copy()
        if population is not None:
            out.remove_nodes_from([node for node in out.nodes if self.data_df.loc[node, "Population"] != population])
        if features is not None:
            out_features = [feature for feature in features if (feature in self.data_df.columns and feature not in ["id", "Population", "Connections", "Train", "Eval"])]
        else:
            out_features = [feature for feature in self.data_df.columns if feature not in ["id", "Population", "Connections", "Train", "Eval"]]

        for feature in out_features:
            nx.set_node_attributes(out, self.data_df[feature].to_dict(), name=feature)
        return out

    def get_graph_torch(self, features: list[str] = None) -> Data:
        """
        Retrieves the data as a PyTorch Geometric Data object.

        Args:
            features (list[str], optional): List of feature columns to include. Defaults to None.

        Returns:
            Data: The data as a PyTorch Geometric Data object.
        """
        data = self.data_df.copy(deep=True)

        edges = []
        for idx, row in data.iterrows():
            for connection in row['Connections']:
                edges.append((idx, connection))
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

        if features is not None:
            out_features = [feature for feature in features if (feature in data.columns and feature not in ["id", "Population", "Connections", "Train", "Eval"])]
        else:
            out_features = [feature for feature in data.columns if feature not in ["id", "Population", "Connections", "Train", "Eval"]]

        x = torch.tensor(data[out_features].values, dtype=torch.float)
        y = torch.tensor(data['Infected'].values, dtype=torch.long)

        num_nodes = data.shape[0]
        indices = torch.randperm(num_nodes)

        train_split = int(0.7 * num_nodes)
        val_split = int(0.85 * num_nodes)

        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        graph = Data(x=x, edge_index=edges, y=y)
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

        return graph

    def optimize(self) -> None:
        """
        Optimizes the data for faster processing.

        Returns:
            None
        """
        self.data_df = pd.concat([self.data_df], axis=1)
