import ast
import os
from typing import Self, Literal

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import pandas as pd


class PopulationData:
    def __init__(self):
        self.data = None
        self.graph = None

    def load(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Self:
        self.data = pd.read_csv(path, index_col=0)
        return self

    def encode_connection_lists(self) -> Self:
        self.data['Connections'] = self.data['Connections'].apply(ast.literal_eval)
        return self

    def encode_population_int(self) -> Self:
        self.data['Population'] = self.data['Population'].apply(lambda x: int(x.replace('A', '')))
        return self

    def encode_degrees(self) -> Self:
        self.data['Degrees'] = self.data['Connections'].apply(len)
        return self

    def encode_mean_neighbor_age(self) -> Self:
        id_to_age = dict(zip(self.data['ID'], self.data['Age']))
        self.data['Mean_Neighbor_Age'] = self.data['Connections'].apply(
            lambda connections: np.mean([id_to_age[connection] for connection in connections]))
        return self

    def encode_mean_neighbor_constitution(self) -> Self:
        id_to_constitution = dict(zip(self.data['ID'], self.data['Constitution']))
        self.data['Mean_Neighbor_Constitution'] = self.data['Connections'].apply(
            lambda connections: np.mean([id_to_constitution[connection] for connection in connections]))
        return self

    def encode_mean_neighbor_behaviour(self) -> Self:
        id_to_behaviour = dict(zip(self.data['ID'], self.data['Behaviour']))
        self.data['Mean_Neighbor_Behaviour'] = self.data['Connections'].apply(
            lambda connections: np.mean([id_to_behaviour[connection] for connection in connections]))
        return self

    def encode_test_train(self) -> Self:
        indices = self.data.index
        train_idx, test_idx = train_test_split(indices, test_size=0.2)
        self.data['Train'] = self.data.index.isin(train_idx)
        self.data['Test'] = self.data.index.isin(test_idx)
        return self

    def drop_population(self) -> Self:
        self.data.drop(columns=['Population'], inplace=True)
        return self

    def drop_index_patient(self) -> Self:
        self.data.drop(columns=['Index_Patient'], inplace=True)
        return self

    def drop_connections(self) -> Self:
        self.data.drop(columns=['Connections'], inplace=True)
        return self

    def drop_degrees(self):
        self.data.drop(columns=['Degrees'], inplace=True)
        return self

    def drop_id(self):
        self.data.drop(columns=['ID'], inplace=True)
        return self

    def get_data(self,
                 train: Literal['train'] | Literal['test'] | Literal['both'] = 'both',
                 x: Literal['x'] | Literal['y'] | Literal['both'] = 'both',
                 population: int = None) -> pd.DataFrame:
        out = self.data.copy(deep=True)
        if train == 'train':
            out = out[out["Train"]]
            out = out.drop(columns=["Train", "Test"])
        elif train == 'test':
            out = out[out["Test"]]
            out = out.drop(columns=["Train", "Test"])
        if x == 'x':
            out = out.drop(columns=["Infected"])
        elif x == 'y':
            out = out["Infected"]
        if population is not None:
            out = out[out["Population"] == population]
            out = out.drop(columns=["Population"])
        return out

    def get_numpy(self,
                  train: Literal['train'] | Literal['test'] | Literal['both'] = 'both',
                  x: Literal['x'] | Literal['y'] | Literal['both'] = 'both',
                  population: int = None) -> np.ndarray:
        out = self.get_data(train=train, x=x, population=population)
        return out.to_numpy()

    def get_graph(self,
                  population: int = None) -> Data:
        if population is not None:
            data = self.data[self.data['Population'] == population].copy(deep=True)
        else:
            data = self.data.copy(deep=True)

        edges = []
        for idx, row in data.iterrows():
            for connection in row['Connections']:
                edges.append((row['ID'], connection))
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.tensor(data[['Age', 'Constitution', 'Behaviour']].values, dtype=torch.float)
        y = torch.tensor(data['Infected'].values, dtype=torch.long)

        train_mask = torch.tensor(data['Train'].values, dtype=torch.bool)
        test_mask = torch.tensor(data['Test'].values, dtype=torch.bool)

        graph_data = Data(x=x, edge_index=edges, y=y)
        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask

        return graph_data

    def get_dataloaders(self,
                        batch_size: int,
                        population: int = None) -> tuple[DataLoader, DataLoader]:
        pass

if __name__ == '__main__':
    data = PopulationData().load("../data/raw/train.csv")
    data.encode_connection_lists().encode_degrees()
    data.drop_population().drop_index_patient().drop_connections().drop_id()
    data.encode_test_train()
    train_x = data.get_numpy(train='train', x='x')
    train_y = data.get_numpy(train='train', x='y')
    test_x = data.get_numpy(train='test', x='x')
    test_y = data.get_numpy(train='test', x='y')
