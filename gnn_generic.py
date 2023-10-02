import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn

# Load the CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Define the Graph Neural Network model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(dataset.num_node_features, 16)
        self.conv2 = pyg_nn.GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        self.embeddings = x.detach()
        x = torch.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.log_softmax(x, dim=1)

# Instantiate the model and optimizer
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset[0])
    loss = torch.nn.functional.nll_loss(out[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss.backward()
    optimizer.step()

    # Test on model.embeddings