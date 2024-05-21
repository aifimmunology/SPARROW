import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,dense
import logging

class minCUTPooling(nn.Module):
    """
    Graph Attention Network (GAT) with minCUT Pooling for microenvironment delineation with node-level attention mechanisms
    Parameters:
        in_dim (int): Input dimension.
        hidden_dims (list of int): Dimensions of hidden layers in the GAT.
        out_dims (list of int): Output dimensions, i.e. the numbers of microenvironment zones
        heads (int): Number of attention heads.
    """
    def __init__(self, in_dim, hidden_dims, out_dims, heads):
        super(minCUTPooling, self).__init__()
        GATconvList=[]
        bnList=[]
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            GATconvList.append(GATConv(prev_dim, hidden_dim, heads=heads, concat=False))
            bnList.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        self.convs=nn.Sequential(*GATconvList)
        self.bn=nn.Sequential(*bnList)
        #self.convs = nn.Sequential(*[GATConv(in_dim if i == 0 else hidden_dim ,
        #                                            hidden_dim, heads=heads, concat=False)
        #                                    for i in range(num_layers)]
        #                                   )
        #self.fc = nn.Linear(hidden_dim , out_dim) 
        fcList=[]
        for out_dim in out_dims:
            fcList.append(nn.Linear(prev_dim , out_dim) )
            fcList.append(nn.BatchNorm1d(out_dim))
            prev_dim = out_dim
        self.fcs = nn.Sequential (*fcList)

    def forward(self, x, edge_index):
        """
        Paramters:
            x (Tensor): the input node features
            edge_index (LongTensor): edge indices in the graph
        Returns:
            list of Tensor: List of outputs from each fully connected layer, series of microenvironment zones at varying resolution
        """
        for conv,bn in zip(self.convs,self.bn):
            x = conv(x, edge_index)
            x=bn(x)
            x=F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        n=0
        outputs=[]
        for fc in self.fcs:    
            x=fc(x)
            if n%2==0 and n!=0:
                x=F.elu(x)
            elif n%2 == 1:
                outputs.append(x)
            n+=1
            
        return outputs

def _loss_smo(embedding,edge_index):
    
    #add a smoothness penalty term to penalize differences between H_i and H_i+1
    delta = embedding[edge_index[0]] - embedding[edge_index[1]]
    return torch.sum (torch.norm (delta,p=2, dim=1))


def train(GATmodel,VAEmodel, optimizer_gat, A_list, X_list ,num_epochs=200,log_name='GATtraining.log', alpha = 1e-6 , **kwargs):
    """
    Training function for the GAT model.

    Parameters:
        GATmodel: SPARROW GAT model instance.
        VAEmodel: SPARROW VAE model (needs to be fully trained).
        optimizer_gat: GAT model optimizer.
        A_list (list): List of adjacency matrices.
        X_list (list): List of VAE latent representation Z tensors.
        num_epochs (int): Number of training epochs.
        log_name (str): Log file name.
        alpha (float): Weight for the smoothness penalty term.
        **kwargs: Additional arguments.
    """
    VAEmodel.eval()
    GATmodel.train()
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
    for epoch in range (num_epochs):
        total_loss=0.0
        for i, (A, X) in enumerate(zip(A_list, X_list)):
            try:
                with torch.no_grad():
                    loc, scale = VAEmodel.Encoder(X)
                optimizer_gat.zero_grad()
                embeddings = GATmodel(loc, A.coalesce().indices())
                loss = 0 
                for embedding in embeddings:
                    x, adj, lc, lo = dense.dense_mincut_pool(loc, A.to_dense(), embedding)
                    loss += (lc + lo + alpha * _loss_smo(embedding,  A.coalesce().indices() )) 
                loss.backward()
                total_loss += loss.item()
                optimizer_gat.step()
            except Exception as e:
                logging.error(f"Error during training on batch {i}: {e}")       
        avg_loss=total_loss/len(A_list)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        