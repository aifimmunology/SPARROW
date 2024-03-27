import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,dense
import logging
class minCUTPooling(nn.Module):
    """
    Graph Attention Network (GAT) with minCUT Pooling for microenvironment delineation
    Parameters:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        out_dim (int): Output dimension, i.e. the number of microenvironment zones
        heads (int): Number of attention heads.
        num_layers (int): Number of GAT layers.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, heads, num_layers):
        super(minCUTPooling, self).__init__()

        self.convs = nn.Sequential(*[GATConv(in_dim if i == 0 else hidden_dim ,
                                                    hidden_dim, heads=heads, concat=False)
                                            for i in range(num_layers)]
                                           )
        self.fc = nn.Linear(hidden_dim , out_dim) 
            


    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x=F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x=self.fc(x)
        return x



def train(GATmodel,VAEmodel, optimizer_gat, A_list, X_list ,num_epochs=200,log_name='GATtraining.log', **kwargs):
    """
    Training function for the GAT model.

    Parameters:
        GATmodel: SPARROW GAT model instance.
        VAEmodel: SPARROW VAE model (needs to be fully trained).
        optimizer_gat: GAT model optimizer.
        A_list (list): List of adjacency matrices.
        X_list (list): List of cell by gene tensors.
        num_epochs (int): Number of training epochs.
        log_name (str): Log file name.
        **kwargs: Additional arguments.
    """
    VAEmodel.eval()
    GATmodel.train()
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
    for epoch in range (num_epochs):
        total_loss=0.0
        for i in range(len(A_list)):
            with torch.no_grad():
                loc,scale=VAEmodel.Encoder(X_list[i])
            optimizer_gat.zero_grad()
            embeddings = GATmodel(loc.detach(), A_list[i].coalesce().indices())  
            x, adj, lc,lo = dense.dense_mincut_pool(loc.detach(),A_list[i].to_dense(),embeddings)
            (lc+lo).backward()
            total_loss+=(lc+lo)
            optimizer_gat.step()
        avg_loss=total_loss/len(A_list)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        