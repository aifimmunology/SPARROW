import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,dense,dense_diff_pool
import logging
import torch.nn.init as init
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

class GNNEdgePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim,out_dim, heads):
        super(GNNEdgePredictor, self).__init__()
        # Transformation layers to standardize feature dimensions
        self.feature_transform = nn.Linear(in_dim, hidden_dim)
        self.link_pred_layer = nn.Bilinear(in_dim, in_dim, 1)
        # GCN layers
        self.conv1 = GATConv(hidden_dim, hidden_dim,heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, coarsened=False):
        x = F.relu(self.feature_transform(x))
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def predict_edges(self, Z):
        # Use cosine similarity to predict edges from embeddings
        n=Z.size(0)
        row, col = torch.combinations(torch.arange(n), 2).t()

        link_logits = self.link_pred_layer(Z[row], Z[col]).squeeze()
        edge_prob_matrix = torch.zeros(n,n)
        edge_prob_matrix[row, col] = torch.sigmoid(link_logits)
        edge_prob_matrix = edge_prob_matrix + edge_prob_matrix.T
        edge_prob_matrix.fill_diagonal_(1)
        return edge_prob_matrix # Edge probability matrix

    
'''
class EmbeddingAndLinkPredictor(nn.Module):
    def __init__(self,in_dim, hidden_dims, latent_dim,embedding_dim,embedding_dims=None):
        super(EmbeddingAndLinkPredictor, self).__init__()
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim


        self.encoder = nn.Sequential(*layers)
        
        
        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[0], in_dim))
        self.decoder = nn.Sequential(*layers)
        
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_layer = nn.Linear(hidden_dims[-1], latent_dim)

        
        # Additional layers to transform z to z' (node embedding in SPARROW GAT)
        layers = []
        prev_dim = latent_dim
        if embedding_dims is not None:
            for embedding_dim_ in embedding_dims:
                layers.append(nn.Linear(prev_dim, embedding_dim_))
                layers.append(nn.BatchNorm1d(embedding_dim_))
                layers.append(nn.ELU())
                prev_dim = embedding_dim_
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.H_prime_to_H = nn.Sequential(*layers)
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
       
    
    def forward(self,x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        H_prime = self.reparameterize(mu, log_var)
        x_prime = self.decoder(H_prime)
        H = self.H_prime_to_H(H_prime)
        #normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        #link_probs = torch.matmul(H_prime, H_prime.t())
        return x_prime,H,mu,log_var
'''            
class GATWithLinkPrediction(minCUTPooling):
    def __init__(self, in_dim, hidden_dims, out_dims, heads,top_k=7):
        super(GATWithLinkPrediction, self).__init__(in_dim, hidden_dims, out_dims, heads)
        # Define initial linear layers for link prediction
        self.initial_linear = nn.Sequential(*[nn.Linear(in_dim, hidden_dims[-1]),nn.BatchNorm1d(hidden_dims[-1]),nn.ELU(),
                                              
                                             nn.Linear(hidden_dims[-1],out_dims[-1]),nn.BatchNorm1d(out_dims[-1]),nn.ELU()])
        self.link_pred_layer = nn.Bilinear(out_dims[-1], out_dims[-1], 1)
        self.top_k=top_k
        #init.kaiming_normal_(self.initial_linear[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        #init.kaiming_normal_(self.initial_linear[3].weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, edge_index=None):
        # Initial link prediction
        if edge_index is None:
            x_initial = self.initial_linear(x)
            edge_index,link_logits = self.predict_links_and_form_edge_index(x_initial,self.top_k)
        else:
            link_logits = None
        # Pass through GAT layers
        embeddings = super().forward(x, edge_index)
        return embeddings,edge_index,link_logits


    def predict_links_and_form_edge_index(self, embeddings, top_k):
        num_nodes = embeddings.size(0)
        row, col = torch.combinations(torch.arange(num_nodes), 2).t()
        link_logits = self.link_pred_layer(embeddings[row], embeddings[col]).squeeze()

        # Convert logits to probabilities
        link_probs = torch.sigmoid(link_logits)

        # Select top N edges for each node
        edge_index = []
        for node_idx in range(num_nodes):
            # Mask to select edges connected to the current node
            mask = (row == node_idx) | (col == node_idx)
            node_probs = link_probs[mask]
            node_edges = torch.stack([row[mask], col[mask]], dim=0)

            # Get indices of top N probabilities
            _, top_indices = torch.topk(node_probs, self.top_k, sorted=False)

            # Select the top N edges
            edge_index.append(node_edges[:, top_indices])

        # Concatenate all top edges and remove duplicates
        edge_index = torch.cat(edge_index, dim=1).unique(dim=1)

        return edge_index, link_logits

    

'''
def train(GATmodel,VAEmodel, optimizer_gat, A_list, X_list ,num_epochs=200,log_name='GATtraining.log',pooling_method='mincut', **kwargs):
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
            if pooling_method=='mincut':
                embeddings = GATmodel(loc.detach(), A_list[i].coalesce().indices())  
                for e in embeddings:
                    x, adj, lc,lo = dense.dense_mincut_pool(loc.detach(),A_list[i].to_dense(),e)
                    (lc+lo).backward()
                    total_loss+=(lc+lo)
                    optimizer_gat.step()
            else: 
                raise ValueError("Invalid pooling method specified. Only mincut is accepted for now.")
        avg_loss=total_loss/len(A_list)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
'''

def train(GATmodel, VAEmodel, optimizer_gat, A_list, X_list, num_epochs=200, log_name='GATtraining.log', method='mincut', **kwargs):
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
        method (str): Method to use ('mincut' or 'linkprediction').
        **kwargs: Additional arguments.
    """
    VAEmodel.eval()
    GATmodel.train()
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(len(A_list)):
            with torch.no_grad():
                loc, _ = VAEmodel.Encoder(X_list[i])

            optimizer_gat.zero_grad()

            if method == 'mincut':
                outputs = GATmodel(loc.detach(), A_list[i].coalesce().indices())

                for e in outputs:
                    x, adj, lc, lo = dense.dense_mincut_pool(loc.detach(), A_list[i].to_dense(), e)
                    ( lc + lo) .backward()
                    total_loss+=(lc+lo).item()
                    
            elif method == 'linkprediction':
                # Link prediction training logic
                embeddings, link_logits = GATmodel(loc.detach())
                for e in embeddings:
                    x, adj, lc, lo = dense.dense_mincut_pool(loc.detach(), A_list[i].to_dense(), e)
                    ( lc + lo) .backward()
                    total_loss+=(lc+lo).item()
                adj_ground_truth = A_list[i].to_dense() > 0
                link_pred_loss = F.binary_cross_entropy_with_logits(link_logits, adj_ground_truth.float())
                link_pred_loss.backward()
                total_loss += link_pred_loss.item()                
            else:
                raise ValueError("Invalid pooling method specified. Only mincut is accepted for now.")
            optimizer_gat.step()

        avg_loss = total_loss / len(A_list)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
