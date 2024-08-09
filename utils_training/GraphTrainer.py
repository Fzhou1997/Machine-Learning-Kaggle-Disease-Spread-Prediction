import torch
from torch_geometric.data import Data, DataLoader
from typing import Literal
from tqdm import tqdm

class GraphTrainer:
    """
    A class to train and evaluate a PyTorch model on graph data.

    Attributes:
        model (Module): The PyTorch model to be trained.
        criterion (Callable): The loss function.
        optimizer (Optimizer): The optimizer for training the model.
    """

    def __init__(self, model, criterion, optimizer):
        """
        Initialize the GraphTrainer with a model, criterion, and optimizer.

        Args:
            model (Module): The PyTorch model to be trained.
            criterion (Callable): The loss function.
            optimizer (Optimizer): The optimizer for training the model.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _train_one_epoch(self, graph_data) -> float:
        """
        Train the model for one epoch.

        Args:
            graph_data (Data): The graph data for training.

        Returns:
            float: The training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
        loss = self.criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask].float().to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval_one_epoch(self, graph_data) -> float:
        """
        Evaluate the model for one epoch.

        Args:
            graph_data (Data): The graph data for evaluation.

        Returns:
            float: The evaluation loss for the epoch.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            loss = self.criterion(out[graph_data.val_mask], graph_data.y[graph_data.val_mask].float().to(self.device))
        return loss.item()

    def train(self, graph_data: Data, num_epochs: int = 100, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        """
        Train the model on the given graph data.

        Args:
            graph_data (Data): The graph data for training.
            num_epochs (int, optional): The number of epochs to train. Default is 100.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.
        """
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
        """
        Test the model on the given graph data.

        Args:
            graph_data (Data): The graph data for testing.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.
        """
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            out = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            loss = self.criterion(out[graph_data.test_mask], graph_data.y[graph_data.test_mask].float().to(self.device))
            print(f'Test loss: {loss:.4f}')