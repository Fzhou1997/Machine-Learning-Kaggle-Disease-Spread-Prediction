import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.nn import GCNConv, GCN
from torch_geometric.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import matplotlib.pyplot as plt


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
    
    def calculate_roc_auc(self, loader, plot=False):
        self.model.eval()
        y_true = []
        y_pred_prob = []

        with torch.no_grad():
            for data in loader:
                out = self.model(data.x, data.edge_index)
                y_true.extend(data.y[data.test_mask].cpu().numpy())
                y_pred_prob.extend(out[data.test_mask][:, 1].cpu().numpy())

        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        roc_auc = roc_auc_score(y_true, y_pred_prob)

        if plot:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

        return roc_auc