import torch
from torch_geometric.data import Data, DataLoader
from typing import Literal
from tqdm import tqdm

class GraphTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _train_one_epoch(self) -> float:
        running_loss = 0
        self.model.train()
        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask].float())
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        running_loss /= len(self.train_loader)
        return running_loss

    def _eval_one_epoch(self) -> float:
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.eval_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index)
                loss += self.criterion(out[data.val_mask], data.y[data.val_mask].float()).item()
        loss /= len(self.eval_loader)
        return loss

    def train(self,
              graph_data: Data,
              batch_size: int = 32,
              num_epochs: int = 100,
              device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        train_mask = graph_data.train_mask
        val_mask = graph_data.val_mask
        train_indices = torch.where(train_mask)[0]
        val_indices = torch.where(val_mask)[0]
        train_data = graph_data.subgraph(train_indices)
        val_data = graph_data.subgraph(val_indices)
        self.train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader([val_data], batch_size=batch_size, shuffle=False)

        self.device = device
        self.model.to(self.device)
        self.train_losses = []
        self.eval_losses = []
        best_eval_loss = float('inf')
        best_model_state = None
        best_model_epoch = -1
        for epoch in tqdm(range(num_epochs)):
            train_loss = self._train_one_epoch()
            eval_loss = self._eval_one_epoch()
            self.train_losses.append(train_loss)
            self.eval_losses.append(eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = self.model.state_dict()
                best_model_epoch = epoch
        print(f'Best model found at epoch {best_model_epoch} with evaluation loss: {best_eval_loss:.4f}')
        self.model.load_state_dict(best_model_state)

    def test(self,
             graph_data: Data,
             batch_size: int = 32,
             device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        test_loader = DataLoader([graph_data], batch_size=batch_size, shuffle=False)
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index)
                out = out.squeeze()
                test_mask = data.test_mask
                loss = self.criterion(out[test_mask], data.y[test_mask].float()).item()
                total_loss += loss
                num_batches += 1
        average_loss = total_loss / num_batches
        print(f'Test loss: {average_loss:.4f}')