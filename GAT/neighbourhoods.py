import torch
from torch_geometric.utils import negative_sampling
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,dense,global_add_pool
import logging


class GATLayerWithSkip(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=False):
        super(GATLayerWithSkip, self).__init__()
        self.gat_conv = GATConv(in_dim, out_dim, heads=heads, concat=concat)
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, x, edge_index):
        identity = x
        x = self.gat_conv(x, edge_index)
        
        if self.res_proj is not None:
            identity = self.res_proj(identity)
        
        x = x + identity
        return x

    
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
            GATconvList.append(GATLayerWithSkip(prev_dim, hidden_dim, heads=heads, concat=False))
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

class DGI(nn.Module):
    def __init__(self, gat_model, mode='self-supervised'):
        super(DGI, self).__init__()
        self.gnn = gat_model
        self.readout = global_add_pool
        self.sigm = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mode = mode
    
    def forward(self, x, edge_index, neg_edge_index):
        pos_z = self.gnn(x, edge_index)
        neg_z = self.gnn(x, neg_edge_index)
        summary = self.sigm(self.readout(pos_z[-1], torch.arange(pos_z[-1].size(0), device=x.device)))
        
        pos_score = (pos_z[-1] * summary).sum(dim=1)
        neg_score = (neg_z[-1] * summary).sum(dim=1)
        
        return pos_score, neg_score 

    def calculate_loss(self, pos_score, neg_score, x=None, edge_index=None):
        y = torch.cat([torch.ones(pos_score.size(0), device=pos_score.device), torch.zeros(neg_score.size(0), device=neg_score.device)])
        scores = torch.cat([pos_score, neg_score])
        dgi_loss = self.bce_loss(scores, y)
        
        if self.mode == 'self-supervised':
            return dgi_loss
        elif self.mode == 'supervised':
            if x is None:
                raise ValueError("Parameter 'x' must be provided.")
            if edge_index is None:
                raise ValueError("Parameter 'edge_index' must be provided.")
            
            pos_z = self.gnn(x, edge_index)
            h, adj, lc, lo = dense.dense_mincut_pool(x, edge_index.to_dense(), pos_z[-1])
            total_loss = dgi_loss + (lc + lo)
            return total_loss, dgi_loss, lc + lo
        

def train(model,VAEmodel, optimizer_gat, A_list, X_list ,patience = 5, num_epochs=200,log_name='GATtraining.log', alpha = 1e-6 , **kwargs):
    """
    Training function for the GAT model.

    Parameters:
        model: SPARROW GAT model instance.
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
    model.train()
    logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(message)s',
                           handlers=[
                    logging.FileHandler(log_name),
                    logging.StreamHandler()])
    prev_dgi_loss = float('inf')
    prev_mincut_loss = float('inf')
    patience_counter = 0    
    for epoch in range (num_epochs):
        prev_dgi_loss_list=[]
        prev_mincut_loss_list=[]
  
        print(f'epoch {epoch}')
        total_loss=0.0
        for i, (A, X) in enumerate(zip(A_list, X_list)):
            try:
                with torch.no_grad():
                    loc, scale = VAEmodel.Encoder(X)
                optimizer_gat.zero_grad()
                neg_edge_index = negative_sampling(A.coalesce().indices(),num_nodes=len(X))
                pos_score, neg_score = model(loc, A.coalesce().indices(), neg_edge_index)
                if model.mode =='supervised':
                    loss, dgi_loss , mincut_loss = model.calculate_loss(pos_score, neg_score , loc, A.coalesce() )
                    
                    prev_dgi_loss_list.append(dgi_loss.item())
                    prev_mincut_loss_list.append(mincut_loss.item())
                    if epoch > 0:
                        #print (dgi_loss_list[i])
                        #print (mincut_loss_list[i])
                        if (dgi_loss.item() > dgi_loss_list[i] and mincut_loss.item() < mincut_loss_list[i]) or (dgi_loss.item() < dgi_loss_list[i] and mincut_loss.item() > mincut_loss_list[i]) or (dgi_loss.item() > dgi_loss_list[i] and mincut_loss.item() > mincut_loss_list[i]):
                            patience_counter+=1
                            if patience_counter >=patience:
                                print(f"Early stopping at epoch {epoch}")
                                logging.info(f"Early stopping at epoch {epoch + 1}")
                                return
                        else:
                            patience_counter = 0
             
                    loss.backward()
                    total_loss += loss.item()
                    print (loss.item(), patience_counter) 
                    optimizer_gat.step()
                    
                else: 
                    dgi_loss = model.calculate_loss(pos_score, neg_score,embedding)
                    dgi_loss.backward()
                    total_loss += dgi_loss.item()
                    optimizer_gat.step()
                    
            except Exception as e:
                print (e)
                logging.error(f"Error during training on batch {i}: {e}")     
        dgi_loss_list = prev_dgi_loss_list.copy()
        mincut_loss_list = prev_mincut_loss_list.copy()
        avg_loss=total_loss/len(A_list)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        