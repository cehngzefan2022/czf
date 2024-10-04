import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_max_pool as gmp

class DeepDNN(nn.Module):
    def __init__(self, input_dim=156, output_dim=1, hidden_dim=2048, num_hidden_layers=8):
        super(DeepDNN, self).__init__()
        
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.Drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, drug1, drug2):
        x1, i1, edge_index1, batch1 = drug1.x, drug1.z, drug1.edge_index, drug1.batch
        x2, i2, edge_index2, batch2 = drug2.x, drug2.z, drug2.edge_index, drug2.batch
        x = torch.cat((x1, x2), 1)
        
        x = self.fc_input(x)
        x = self.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
        x = self.Drop(x)    
        x = self.fc_output(x)
        x = self.sigmoid(x)                               
        return x

