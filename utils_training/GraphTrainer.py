import torch
from torch_geometric.data import Data, DataLoader
from typing import Literal
from tqdm import tqdm

class GraphTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _train_one_epoch(self, graph_data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
        loss = self.criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask].float().to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval_one_epoch(self, graph_data) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            loss = self.criterion(out[graph_data.val_mask], graph_data.y[graph_data.val_mask].float().to(self.device))
        return loss.item()

    def train(self, graph_data: Data, num_epochs: int = 100, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        self.device = device
        self.model.to(self.device)
        best_eval_loss = float('inf')
        best_model_state = None

        for epoch in tqdm(range(num_epochs)):
            train_loss = self._train_one_epoch(graph_data)
            eval_loss = self._eval_one_epoch(graph_data)
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = self.model.state_dict()
            
        print(f'Best model found with evaluation loss: {best_eval_loss:.4f}')
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def test(self, graph_data: Data, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            loss = self.criterion(out[graph_data.test_mask], graph_data.y[graph_data.test_mask].float().to(self.device))
            print(f'Test loss: {loss:.4f}')