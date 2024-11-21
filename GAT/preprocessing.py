#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import libpysal
import scipy.sparse
def create_sparse_adj_matrix(matrix):
    # Get the non-zero indices and values of the input matrix
    row, col = scipy.sparse.csr_matrix(matrix).nonzero()
    
    # Create a new sparse matrix with ones at non-zero locations
    mat = scipy.sparse.csr_matrix((np.ones_like(row), (row, col)), shape=matrix.shape)
    mat = add_self_loop(mat)
    return mat

def add_self_loop(matrix):
    matrix.setdiag(1)
    return matrix

def normalize_adjacency(adjacency_matrix, degree_matrix):
    # Calculate the inverse degree matrix
    inv_degree_matrix = scipy.sparse.linalg.inv(degree_matrix)
    
    # Normalize the adjacency matrix using D^-1A
    normalized_adjacency = inv_degree_matrix.dot(adjacency_matrix)
    
    return normalized_adjacency


def compute_sparse_degree_matrix(adjacency_matrix):
    # Compute the in-degree vector by summing the columns of the adjacency matrix
    in_degree = np.array(adjacency_matrix.sum(axis=0)).flatten()
    
    # Compute the out-degree vector by summing the rows of the adjacency matrix
    out_degree = np.array(adjacency_matrix.sum(axis=1)).flatten()
    
    # Create a diagonal matrix from the in-degree and out-degree vectors
    degree_matrix = scipy.sparse.diags(in_degree+ out_degree,shape=adjacency_matrix.shape,format='csr')
    
    return degree_matrix

def convert_scipy_csr_to_sparse_tensor(scipy_csr):
    # Convert the CSR matrix to COO format
    coo_matrix = scipy_csr.tocoo()
    
    # Convert the COO matrix to a sparse PyTorch tensor
    indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
    values = torch.FloatTensor(coo_matrix.data)
    shape = torch.Size(coo_matrix.shape)
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    
    return sparse_tensor

def calculate_w(geopandas_df,w_method='knn',**kwargs):
    '''
    calculate spatial weight matrix for a GeoDataFrame
    Parameters:
        geopandas_df (geopandas.GeoDataFrame) : GeoDataFrame containing spatial coordinates.

        w_method (str): weight matrix calculation method {'knn','DistanceBand','kernel'}, default='knn' see https://pysal.org/libpysal/api.html#distance-weights for details
        
        **kwargs: Additional keyword arguments for weight matrix calculation. They need to be arguments recognised by libpysal
        
    Returns 
     
         libpysal.weights.W: Spatial weights matrix.
    '''
    assert w_method in {'knn', 'DistanceBand', 'kernel'}, "Invalid w_method. Choose from 'knn', 'DistanceBand', 'kernel'."
    default_params = {
        'knn': {'k': 7},
        'DistanceBand': {'threshold': 200.0},
        'kernel': {'function': 'Gaussian'}
    }
    params=default_params[w_method]
    params.update(kwargs)
    if w_method=='knn':
        w=libpysal.weights.KNN.from_dataframe(geopandas_df,**params)
    elif w_method=='DistanceBand':
        w=libpysal.weights.DistanceBand.from_dataframe(geopandas_df,**params)
    elif w_method=='kernel':     
        w=libpysal.weights.Kernel(geopandas_df,**params)
    w.transform='R'
    return(w)

def _make_A(gdf,k,w_method='knn'):
    '''
    Parameters:
        gdf:  geopandas.geodataframe.GeoDataFrame
            dataframe containing a geometry column
        k_neighbours: int
            Number of neighbors for spatial weight calculation.
        w_method: str, optional
            Method to calculate weights; defaults to 'knn'.

    Returns:
        A:torch.sparse.FloatTensor
        Normalized adjacency matrix.
        
    '''
    spatial_weight=calculate_w(gdf,
                               w_method='knn',
                               k=k)
    _sparse_adj_mat=create_sparse_adj_matrix(spatial_weight.sparse)
    D=compute_sparse_degree_matrix(_sparse_adj_mat)
    A=normalize_adjacency(_sparse_adj_mat,D)
    A=convert_scipy_csr_to_sparse_tensor(A)
    return A

def graph_loader(obj,genes,x0,x1,y0,y1,stepsize,k_neighbours,min_sum=5, **kwargs):
    '''
    a graph loader for SPARROW GAT 
    Parameters:
        obj : 
            SPARROW 
        genes : list or np.ndarray or str  
            selected feature genes from SPARROW VAE. It can be a list or array of gene names or a file name string that 
            points to a file containing gene names
        min_sum: int
            minimum transcript sum threshold for selected training cells
        x0,x1,y0,y1: int
            bounding box coordinates for spatial regions chosen for training
        stepsize:
            step size for tiling over the spatial area
        k_neighbours: int
            Number of neighbors for spatial weight calculation.
        min_sum: int, optional
            Minimum sum of transcripts threshold for selecting training cells; defaults to 5.
        **kwargs: Additional keyword arguments.
            
    Returns:
        A_list: 
            list of normalised adjacency matrices
        X_list:list of torch.Tensor
            list of cell by gene tensors to be fed into SPARROW VAE to get latent representations Z for node features.
            
    '''
    if isinstance(genes, str):
        genes=np.genfromtxt(genes,dtype='str')
    else:
        assert isinstance(genes,(list, np.ndarray)), "Invalid type. Provide a list or numpy array of selected feature genes from SPARROW VAE."
    A_list=[]
    X_list=[]
    for i in range(0,x1-x0,stepsize):
        for j in range(0,y1-y0,stepsize):
            selected_training_cells=obj.parquet.compute().loc[obj.geometry.cx[x0+i:x0+i+stepsize,
                                                                              y0+j:y0+j+stepsize].index,genes]
            selected_training_cells=selected_training_cells[selected_training_cells.sum(axis=1) >=min_sum]
            
            if len(selected_training_cells) > kwargs.get('min_length',3000): #this is an arbitrary number just so edges with disturbed structures aren't included in training
                X_list.append(torch.from_numpy(selected_training_cells.values).type(torch.float))
                print (selected_training_cells.shape)
                A=_make_A(obj.geometry.loc[ selected_training_cells.index].set_index(np.arange(len(selected_training_cells))), k_neighbours, kwargs.get('w_method','knn'))
                A_list.append( A)
    return A_list,X_list
