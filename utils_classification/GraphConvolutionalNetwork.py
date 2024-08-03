import torch
from torch_geometric.nn import GCNConv, GCN
from torch_geometric.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class GraphConvolutionalNetwork:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.model = GCN(in_channels, hidden_channels, out_channels)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

    def fit(self, loader, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data in loader:
                self.optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = self.loss_fn(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
            self.scheduler.step()

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in loader:
                out = self.model(data.x, data.edge_index)
                loss = self.loss_fn(out[data.test_mask], data.y[data.test_mask])
                total_loss += loss.item()
                pred = out[data.test_mask].argmax(dim=1)
                correct += pred.eq(data.y[data.test_mask]).sum().item()
                total_samples += data.test_mask.sum().item()
        accuracy = correct / total_samples
        print(f'Test Loss: {total_loss/len(loader):.4f}, Accuracy: {accuracy:.4f}')
        return total_loss / len(loader), accuracy