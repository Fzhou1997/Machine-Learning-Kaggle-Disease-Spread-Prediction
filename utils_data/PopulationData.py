import ast
import os
from typing import Self, Literal

import numpy as np
import networkx as nx
import torch
from node2vec import Node2Vec
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import Data, DataLoader

import pandas as pd
from torch_geometric.utils import from_networkx


class PopulationData:
    def __init__(self):
        self.data_df = None
        self.graph_nx = None

    def load_raw(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Self:
        self.data_df = pd.read_csv(path, index_col='ID')
        self.data_df.drop(columns=['id'], inplace=True)
        return self

    def load_processed(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Self:
        self.data_df = pd.read_csv(path, index_col='ID')
        return self

    def save_processed(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        self.data_df.to_csv(path)

    def encode_normalized_age(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Age'] = scaler.fit_transform(self.data_df[['Age']])
        return self

    def encode_normalized_behavior(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Behaviour'] = scaler.fit_transform(self.data_df[['Behaviour']])
        return self

    def encode_normalized_constitution(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Constitution'] = scaler.fit_transform(self.data_df[['Constitution']])
        return self

    def encode_standardized_age(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Age'] = scaler.fit_transform(self.data_df[['Age']])
        return self

    def encode_standardized_constitution(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Constitution'] = scaler.fit_transform(self.data_df[['Constitution']])
        return self

    def encode_standardized_behavior(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Behaviour'] = scaler.fit_transform(self.data_df[['Behaviour']])
        return self

    def encode_connection_lists(self) -> Self:
        self.data_df['Connections'] = self.data_df['Connections'].apply(ast.literal_eval)
        return self

    def encode_graph_nx(self) -> Self:
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
        degree_dict = dict(self.graph_nx.degree())
        degree_series = pd.Series(degree_dict)
        self.data_df['Degree'] = degree_series
        return

    def encode_degree_centrality(self) -> Self:
        degree_centrality_dict = nx.degree_centrality(self.graph_nx)
        degree_centrality_series = pd.Series(degree_centrality_dict)
        self.data_df['Degree_Centrality'] = degree_centrality_series
        return self

    def encode_clustering_coefficient(self) -> Self:
        clustering_coefficient_dict = nx.clustering(self.graph_nx)
        clustering_coefficient_series = pd.Series(clustering_coefficient_dict)
        self.data_df['Clustering_Coefficient'] = clustering_coefficient_series
        return self

    def encode_normalized_degree(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Degree'] = scaler.fit_transform(self.data_df[['Degree']])
        return self

    def encode_normalized_degree_centrality(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Degree_Centrality']])
        return self

    def encode_normalized_clustering_coefficient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Clustering_Coefficient']])
        return self

    def encode_standardized_degree(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Degree'] = scaler.fit_transform(self.data_df[['Degree']])
        return self

    def encode_standardized_degree_centrality(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Degree_Centrality']])
        return self

    def encode_standardized_clustering_coefficient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Clustering_Coefficient']])
        return self

    def encode_connected_index_patient(self) -> Self:
        index_patients = self.data_df[self.data_df['Index_Patient'] == 1]
        index_patients_dict = dict(zip(index_patients['Population'], index_patients.index))
        self.data_df['Connected_Index_Patient'] = self.data_df.apply(
            lambda row: index_patients_dict[row['Population']], axis=1
        )
        return self

    def encode_distance_to_index_patient(self) -> Self:
        index_patients = self.data_df[self.data_df['Index_Patient'] == 1].index.to_list()
        shortest_paths_all = nx.multi_source_dijkstra_path_length(self.graph_nx, index_patients)
        shortest_paths = {node: float('inf') for node in self.data_df.index}
        for node, length in shortest_paths_all.items():
            shortest_paths[node] = length
        self.data_df['Distance_to_Index_Patient'] = pd.Series(shortest_paths)
        return self

    def encode_normalized_distance_to_index_patient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Distance_to_Index_Patient']])
        return self

    def encode_standardized_distance_to_index_patient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Distance_to_Index_Patient']])
        return self

    def encode_sum_neighbor_age(self) -> Self:
        ages = self.data_df['Age'].to_dict()
        self.data_df['Sum_Neighbor_Age'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([ages[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_constitution(self) -> Self:
        constitutions = self.data_df['Constitution'].to_dict()
        self.data_df['Sum_Neighbor_Constitution'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([constitutions[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_behavior(self) -> Self:
        behaviors = self.data_df['Behaviour'].to_dict()
        self.data_df['Sum_Neighbor_Behaviour'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([behaviors[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_degree(self) -> Self:
        degree = self.data_df['Degree'].to_dict()
        self.data_df['Sum_Neighbor_Degree'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([degree[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_degree_centrality(self) -> Self:
        degree_centrality = self.data_df['Degree_Centrality'].to_dict()
        self.data_df['Sum_Neighbor_Degree_Centrality'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([degree_centrality[connection] for connection in connections])
        )
        return self

    def encode_sum_neighbor_clustering_coefficient(self) -> Self:
        clustering_coefficient = self.data_df['Clustering_Coefficient'].to_dict()
        self.data_df['Sum_Neighbor_Clustering_Coefficient'] = self.data_df['Connections'].apply(
            lambda connections: np.sum([clustering_coefficient[connection] for connection in connections])
        )
        return self

    def encode_normalized_sum_neighbor_age(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Age']])
        return self

    def encode_normalized_sum_neighbor_constitution(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Constitution']])
        return self

    def encode_normalized_sum_neighbor_behavior(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Behaviour']])
        return self

    def encode_normalized_sum_neighbor_degree(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree']])
        return self

    def encode_normalized_sum_neighbor_degree_centrality(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree_Centrality']])
        return self

    def encode_normalized_sum_neighbor_clustering_coefficient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Clustering_Coefficient']])
        return self

    def encode_standardized_sum_neighbor_age(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Age']])
        return self

    def encode_standardized_sum_neighbor_constitution(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Constitution']])
        return self

    def encode_standardized_sum_neighbor_behavior(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Behaviour']])
        return self

    def encode_standardized_sum_neighbor_degree(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree']])
        return self

    def encode_standardized_sum_neighbor_degree_centrality(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Degree_Centrality']])
        return self

    def encode_standardized_sum_neighbor_clustering_coefficient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Neighbor_Clustering_Coefficient']])
        return self

    def encode_mean_neighbor_age(self) -> Self:
        ages = self.data_df['Age'].to_dict()
        self.data_df['Mean_Neighbor_Age'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([ages[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_constitution(self) -> Self:
        constitutions = self.data_df['Constitution'].to_dict()
        self.data_df['Mean_Neighbor_Constitution'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([constitutions[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_behavior(self) -> Self:
        behaviors = self.data_df['Behaviour'].to_dict()
        self.data_df['Mean_Neighbor_Behaviour'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([behaviors[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_degree(self) -> Self:
        degree = self.data_df['Degree'].to_dict()
        self.data_df['Mean_Neighbor_Degree'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([degree[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_degree_centrality(self) -> Self:
        degree_centrality = self.data_df['Degree_Centrality'].to_dict()
        self.data_df['Mean_Neighbor_Degree_Centrality'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([degree_centrality[connection] for connection in connections])
        )
        return self

    def encode_mean_neighbor_clustering_coefficient(self) -> Self:
        clustering_coefficient = self.data_df['Clustering_Coefficient'].to_dict()
        self.data_df['Mean_Neighbor_Clustering_Coefficient'] = self.data_df['Connections'].apply(
            lambda connections: np.mean([clustering_coefficient[connection] for connection in connections])
        )
        return self

    def encode_normalized_mean_neighbor_age(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Age']])
        return self

    def encode_normalized_mean_neighbor_constitution(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Constitution']])
        return self

    def encode_normalized_mean_neighbor_behavior(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Behaviour']])
        return self

    def encode_normalized_mean_neighbor_degree(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree']])
        return self

    def encode_normalized_mean_neighbor_degree_centrality(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree_Centrality']])
        return self

    def encode_normalized_mean_neighbor_clustering_coefficient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Clustering_Coefficient']])
        return self

    def encode_standardized_mean_neighbor_age(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Age'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Age']])
        return self

    def encode_standardized_mean_neighbor_constitution(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Constitution']])
        return self

    def encode_standardized_mean_neighbor_behavior(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Behaviour']])
        return self

    def encode_standardized_mean_neighbor_degree(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Degree'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree']])
        return self

    def encode_standardized_mean_neighbor_degree_centrality(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Degree_Centrality']])
        return self

    def encode_standardized_mean_neighbor_clustering_coefficient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Neighbor_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Neighbor_Clustering_Coefficient']])
        return self

    def encode_sum_population_age(self) -> Self:
        sum_population_age = self.data_df.groupby('Population')['Age'].sum()
        self.data_df['Sum_Population_Age'] = self.data_df['Population'].map(sum_population_age)
        return self

    def encode_sum_population_constitution(self) -> Self:
        sum_population_constitution = self.data_df.groupby('Population')['Constitution'].sum()
        self.data_df['Sum_Population_Constitution'] = self.data_df['Population'].map(sum_population_constitution)
        return self

    def encode_sum_population_behavior(self) -> Self:
        sum_population_behavior = self.data_df.groupby('Population')['Behaviour'].sum()
        self.data_df['Sum_Population_Behaviour'] = self.data_df['Population'].map(sum_population_behavior)
        return self

    def encode_sum_population_degree(self) -> Self:
        sum_population_degree = self.data_df.groupby('Population')['Degree'].sum()
        self.data_df['Sum_Population_Degree'] = self.data_df['Population'].map(sum_population_degree)
        return self

    def encode_sum_population_degree_centrality(self) -> Self:
        sum_population_degree_centrality = self.data_df.groupby('Population')['Degree_Centrality'].sum()
        self.data_df['Sum_Population_Degree_Centrality'] = self.data_df['Population'].map(sum_population_degree_centrality)
        return self

    def encode_sum_population_clustering_coefficient(self) -> Self:
        sum_population_clustering_coefficient = self.data_df.groupby('Population')['Clustering_Coefficient'].sum()
        self.data_df['Sum_Population_Clustering_Coefficient'] = self.data_df['Population'].map(sum_population_clustering_coefficient)
        return self

    def encode_normalized_sum_population_age(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Age'] = scaler.fit_transform(self.data_df[['Sum_Population_Age']])
        return self

    def encode_normalized_sum_population_constitution(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Population_Constitution']])
        return self

    def encode_normalized_sum_population_behavior(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Population_Behaviour']])
        return self

    def encode_normalized_sum_population_degree(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Degree'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree']])
        return self

    def encode_normalized_sum_population_degree_centrality(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree_Centrality']])
        return self

    def encode_normalized_sum_population_clustering_coefficient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Population_Clustering_Coefficient']])
        return self

    def encode_standardized_sum_population_age(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Age'] = scaler.fit_transform(self.data_df[['Sum_Population_Age']])
        return self

    def encode_standardized_sum_population_constitution(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Constitution'] = scaler.fit_transform(self.data_df[['Sum_Population_Constitution']])
        return self

    def encode_standardized_sum_population_behavior(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Sum_Population_Behaviour']])
        return self

    def encode_standardized_sum_population_degree(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Degree'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree']])
        return self

    def encode_standardized_sum_population_degree_centrality(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Sum_Population_Degree_Centrality']])
        return self

    def encode_standardized_sum_population_clustering_coefficient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Sum_Population_Clustering_Coefficient']])
        return self

    def encode_mean_population_age(self) -> Self:
        mean_population_age = self.data_df.groupby('Population')['Age'].mean()
        self.data_df['Mean_Population_Age'] = self.data_df['Population'].map(mean_population_age)
        return self

    def encode_mean_population_constitution(self) -> Self:
        mean_population_constitution = self.data_df.groupby('Population')['Constitution'].mean()
        self.data_df['Mean_Population_Constitution'] = self.data_df['Population'].map(mean_population_constitution)
        return self

    def encode_mean_population_behavior(self) -> Self:
        mean_population_behavior = self.data_df.groupby('Population')['Behaviour'].mean()
        self.data_df['Mean_Population_Behaviour'] = self.data_df['Population'].map(mean_population_behavior)
        return self

    def encode_mean_population_degree(self) -> Self:
        mean_population_degree = self.data_df.groupby('Population')['Degree'].mean()
        self.data_df['Mean_Population_Degree'] = self.data_df['Population'].map(mean_population_degree)
        return self

    def encode_mean_population_degree_centrality(self) -> Self:
        mean_population_degree_centrality = self.data_df.groupby('Population')['Degree_Centrality'].mean()
        self.data_df['Mean_Population_Degree_Centrality'] = self.data_df['Population'].map(mean_population_degree_centrality)
        return self

    def encode_mean_population_clustering_coefficient(self) -> Self:
        mean_population_clustering_coefficient = self.data_df.groupby('Population')['Clustering_Coefficient'].mean()
        self.data_df['Mean_Population_Clustering_Coefficient'] = self.data_df['Population'].map(mean_population_clustering_coefficient)
        return self

    def encode_normalized_mean_population_age(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Age'] = scaler.fit_transform(self.data_df[['Mean_Population_Age']])
        return self
    
    def encode_normalized_mean_population_constitution(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Population_Constitution']])
        return self
    
    def encode_normalized_mean_population_behavior(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Population_Behaviour']])
        return self
    
    def encode_normalized_mean_population_degree(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Degree'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree']])
        return self
    
    def encode_normalized_mean_population_degree_centrality(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree_Centrality']])
        return self
    
    def encode_normalized_mean_population_clustering_coefficient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Population_Clustering_Coefficient']])
        return self
    
    def encode_standardized_mean_population_age(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Age'] = scaler.fit_transform(self.data_df[['Mean_Population_Age']])
        return self
    
    def encode_standardized_mean_population_constitution(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Constitution'] = scaler.fit_transform(self.data_df[['Mean_Population_Constitution']])
        return self
    
    def encode_standardized_mean_population_behavior(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Behaviour'] = scaler.fit_transform(self.data_df[['Mean_Population_Behaviour']])
        return self
    
    def encode_standardized_mean_population_degree(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Degree'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree']])
        return self
    
    def encode_standardized_mean_population_degree_centrality(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Degree_Centrality'] = scaler.fit_transform(self.data_df[['Mean_Population_Degree_Centrality']])
        return self
    
    def encode_standardized_mean_population_clustering_coefficient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Clustering_Coefficient'] = scaler.fit_transform(self.data_df[['Mean_Population_Clustering_Coefficient']])
        return

    def encode_sum_population_distance_to_index_patient(self) -> Self:
        sum_population_distance_to_index_patient = self.data_df.groupby('Population')['Distance_to_Index_Patient'].sum()
        self.data_df['Sum_Population_Distance_to_Index_Patient'] = self.data_df['Population'].map(sum_population_distance_to_index_patient)
        return self

    def encode_normalized_sum_population_distance_to_index_patient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Sum_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Sum_Population_Distance_to_Index_Patient']])
        return self

    def encode_standardized_sum_population_distance_to_index_patient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Sum_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Sum_Population_Distance_to_Index_Patient']])
        return self

    def encode_mean_population_distance_to_index_patient(self) -> Self:
        mean_population_distance_to_index_patient = self.data_df.groupby('Population')['Distance_to_Index_Patient'].mean()
        self.data_df['Mean_Population_Distance_to_Index_Patient'] = self.data_df['Population'].map(mean_population_distance_to_index_patient)
        return self

    def encode_normalized_mean_population_distance_to_index_patient(self) -> Self:
        scaler = MinMaxScaler()
        self.data_df['Normalized_Mean_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Mean_Population_Distance_to_Index_Patient']])
        return self

    def encode_standardized_mean_population_distance_to_index_patient(self) -> Self:
        scaler = StandardScaler()
        self.data_df['Standardized_Mean_Population_Distance_to_Index_Patient'] = scaler.fit_transform(self.data_df[['Mean_Population_Distance_to_Index_Patient']])
        return self

    def encode_test_train(self) -> Self:
        indices = self.data_df.index
        train_idx, test_idx = train_test_split(indices, test_size=0.2)
        self.data_df['Train'] = self.data_df.index.isin(train_idx)
        self.data_df['Test'] = self.data_df.index.isin(test_idx)
        return self

    def get_data_dataframes(self,
                            features: list[str] = None,
                            train: Literal['Train', 'Test'] = None,
                            population: str = None) -> tuple[pd.DataFrame, pd.Series]:
        out = self.data_df
        if population is not None:
            out = out[out["Population"] == population]
        if train is not None:
            out = out[out[train]]
        out = out.drop(columns=["Population", "Connections", "Train", "Test"])
        out_features = out.drop(columns=["Infected"])
        out_labels = out["Infected"]
        if features is not None:
            out_columns = [feature for feature in features if feature in out_features.columns]
            out_features = out_features[out_columns]
        return out_features, out_labels

    def get_data_numpy(self,
                       features: list[str] = None,
                       train: Literal['Train', 'Test'] = None,
                       population: str = None) -> tuple[np.ndarray, np.ndarray]:
        features_df, labels_df = self.get_data_dataframes(features=features, train=train, population=population)
        return features_df.to_numpy(), labels_df.to_numpy()

    def get_data_tensors(self,
                         features: list[str] = None,
                         train: Literal['Train', 'Test'] = None,
                         population: int = None) -> tuple[Tensor, Tensor]:
        features_np, labels_np = self.get_data_numpy(features=features, train=train, population=population)
        return torch.tensor(features_np, dtype=torch.float32), torch.tensor(labels_np, dtype=torch.float32)

    def get_graph_nx(self,
                     features: list[str] = None,
                     population: int = None) -> nx.Graph:
        out = self.graph_nx.copy()
        if population is not None:
            out.remove_nodes_from([node for node in out.nodes if self.data_df.loc[node, "Population"] != population])
        out_features = [feature for feature in features if (feature in self.data_df.columns and feature not in ["Population", "Connections", "Train", "Test"])]
        for feature in out_features:
            nx.set_node_attributes(out, self.data_df[feature].to_dict(), name=feature)
        return out

    def get_graph_torch(self,
                        features: list[str] = None,
                        population: int = None) -> Data:
        graph_nx = self.get_graph_nx(features=features, population=population)
        graph_torch = from_networkx(graph_nx)
        return graph_torch
