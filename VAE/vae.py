# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from torch.distributions.utils import probs_to_logits
import logging
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
import os
import json
import numpy as np
from ..preprocessing import ingest
from ..utils import dataloader


#clamping for numerical stability
epsilon = 1e-9
clamp_min = probs_to_logits(torch.tensor(torch.finfo(torch.float32).tiny, dtype=torch.float32))
clamp_max=torch.tensor(16.00)

import torch




class vae2(nn.Module):

    def __init__(self, K, M, hidden_dim_vec_K , hidden_dim_vec, logits):
        """
        Variational Autoencoder (VAE) for cell type inference in spatial transcriptomics.

        Parameters:
            K (int): Dimensionality of meaningful gene features in the scRNA-seq other than those that are included in the gene panel.
            M (int): Dimensionality for a subset of genes or reduced gene panel.
            hidden_dim_vec_K (list of int): Hidden layer dimensions for the encoder and decoder for X'.
            hidden_dim_vec (list of int): Hidden layer dimensions for the encoder and decoder.
            logits (list of int): Dimensions for cell type logits.
        """
        assert hidden_dim_vec_K[-1]  ==  hidden_dim_vec[-1]
        self.K = K
        self.M = M
        self.latent_dim=hidden_dim_vec[-1]
        self.celltype_fine=logits[-1]
        super().__init__()
        
        
        #encoding X as latent representation Z
        encoding_layersK=[nn.Linear(self.K,hidden_dim_vec_K[0]),nn.Dropout(p=0.1), nn.BatchNorm1d(hidden_dim_vec_K[0]),nn.ReLU()]
        for dim1, dim2 in zip (hidden_dim_vec_K[:-1], hidden_dim_vec_K[1:-1] ) : 
            encoding_layersK.append(nn.Linear(dim1,dim2))
            encoding_layersK.append(nn.Dropout(p=0.1))
            encoding_layersK.append(nn.BatchNorm1d( dim2))
            encoding_layersK.append(nn.ReLU())  
        encoding_layersK.append(nn.Linear(hidden_dim_vec_K[-2],hidden_dim_vec_K[-1] * 2))
        encoding_layersK.append(nn.Dropout(p=0.1))
        encoding_layersK.append(nn.BatchNorm1d( hidden_dim_vec_K[-1] * 2))
        self.encoderK = nn.Sequential(*encoding_layersK)
        
        #encoding X as latent representation Z
        encoding_layers=[nn.Linear(M,hidden_dim_vec[0]),nn.Dropout(p=0.1), nn.BatchNorm1d(hidden_dim_vec[0]),nn.ReLU()]
        for dim1, dim2 in zip (hidden_dim_vec[:-1], hidden_dim_vec[1:-1] ) : 
            encoding_layers.append(nn.Linear(dim1,dim2))
            encoding_layers.append(nn.Dropout(p=0.1))
            encoding_layers.append(nn.BatchNorm1d( dim2))
            encoding_layers.append(nn.ReLU())  
        encoding_layers.append(nn.Linear(hidden_dim_vec[-2],hidden_dim_vec[-1] * 2))
        encoding_layers.append(nn.Dropout(p=0.1))
        encoding_layers.append(nn.BatchNorm1d( hidden_dim_vec[-1] * 2))
        self.encoder = nn.Sequential(*encoding_layers)
        
        #decoding X_hat 
        decoding_layers=[]
        for dim1, dim2 in zip (hidden_dim_vec[::-1], hidden_dim_vec[::-1][1:] ) : 
            decoding_layers.append(nn.Linear(dim1,dim2))
            decoding_layers.append(nn.Dropout(p=0.1))
            decoding_layers.append(nn.BatchNorm1d( dim2))
            decoding_layers.append(nn.ReLU())        
        decoding_layers.append(nn.Linear(hidden_dim_vec[0],M*2))
        decoding_layers.append(nn.Dropout(p=0.1))
        decoding_layers.append(nn.BatchNorm1d( M*2))
        self.decoder=nn.Sequential(*decoding_layers)
        
        #decoding X'_hat 
        decoding_layersK=[]
        for dim1, dim2 in zip (hidden_dim_vec_K[::-1], hidden_dim_vec_K[::-1][1:] ) : 
            decoding_layersK.append(nn.Linear(dim1,dim2))
            decoding_layersK.append(nn.Dropout(p=0.1))
            decoding_layersK.append(nn.BatchNorm1d( dim2))
            decoding_layersK.append(nn.ReLU())        
        decoding_layersK.append(nn.Linear(hidden_dim_vec_K[0],K*2))
        decoding_layersK.append(nn.Dropout(p=0.1))
        decoding_layersK.append(nn.BatchNorm1d( K*2))
        self.decoderK=nn.Sequential(*decoding_layersK)
        
        #representing Z as cell type logits
        encoding_logit_layers=[nn.Linear(hidden_dim_vec[-1],logits[0]),nn.Dropout(p=0.1), nn.BatchNorm1d(logits[0]),nn.ReLU()]
        for dim1, dim2 in zip(logits[:-1],logits[1:-1]):
            encoding_logit_layers.append(nn.Linear(dim1,dim2))
            encoding_logit_layers.append(nn.Dropout(p=0.1))
            encoding_logit_layers.append(nn.BatchNorm1d( dim2))
            encoding_logit_layers.append(nn.ReLU())
        encoding_logit_layers.append(nn.Linear(logits[-2],logits[-1]*2))
        encoding_logit_layers.append(nn.Dropout(p=0.1))
        encoding_logit_layers.append(nn.BatchNorm1d(logits[-1]*2))
        self.encoding_logits=nn.Sequential(*encoding_logit_layers)
        
        #decoding cell type logits as Z
        decoding_logits_layers=[]
        for dim1, dim2 in zip(logits[::-1],logits[::-1][1:]):
            decoding_logits_layers.append(nn.Linear(dim1,dim2))
            decoding_logits_layers.append(nn.Dropout(p=0.1))
            decoding_logits_layers.append(nn.BatchNorm1d( dim2))
            decoding_logits_layers.append(nn.ReLU())
        decoding_logits_layers.append(nn.Linear(logits[0],hidden_dim_vec[-1]*2))
        decoding_logits_layers.append(nn.Dropout(p=0.1))
        decoding_logits_layers.append(nn.BatchNorm1d( hidden_dim_vec[-1]*2))
        self.decoding_logits=nn.Sequential(*decoding_logits_layers)
        
    def Encoder(self, X, useK= False):
        """
        Encoder function for the mean and scale of the latent representation Z.

        Parameters:
            X (torch.Tensor): Input cell by gene matrix.
    
        Returns:
            tuple: Mean and scale of the latent representation.

        """
        if useK:
            encoded=self.encoderK(X)
            
        else:
            encoded=self.encoder(X)
        loc,scale=torch.split(encoded,encoded.shape[1]//2,dim=-1)
        scale=softplus(scale) +epsilon
        return loc, scale
    
    def Decoder(self, X,useK=False):
        """
        Decoder function for parameterising X_hat from the latent representation.
        
        Parameters:
            X (torch.Tensor): Latent representation.
            useK (bool): Flag to use decoderK for additional decoding for scRNA-seq, defaults to False.
        
        Returns:
            tuple: logits to parameterisze zero inflated negative binomial distr.
        
        """
        decoded=self.decoder(X)
        gate_logits,nb_logits=torch.split(decoded,decoded.shape[1]//2,dim=-1)
        gate_logits=torch.clamp(gate_logits,min=clamp_min,max=clamp_max)
        nb_logits=torch.clamp(nb_logits,min=clamp_min,max=clamp_max)
        if useK:
            decodedK=self.decoderK(X)
            gate_logitsK,nb_logitsK=torch.split(decodedK,decodedK.shape[1]//2,dim=-1)
            gate_logitsK=torch.clamp(gate_logitsK,min=clamp_min,max=clamp_max)
            nb_logitsK=torch.clamp(nb_logitsK,min=clamp_min,max=clamp_max)
            gate_logits = gate_logitsK
            nb_logits = nb_logitsK
        
        return gate_logits, nb_logits
    
    def Encoder_logits(self,X):
        
        encoded=self.encoding_logits(X)
        loc,scale=torch.split(encoded,encoded.shape[1]//2,dim=-1)
        scale=softplus(scale) + epsilon
        return loc, scale
    def Decoder_logits(self,X):
        decoded=self.decoding_logits(X)
        loc,scale=torch.split(decoded,decoded.shape[1]//2,dim=-1)
        scale=softplus(scale) +epsilon
        return loc, scale
    
    #posterior q(z|x)
    def guide (self,X=None,X_prime=None,L=None,class_weights=None):
        pyro.module("vae",self)
        if X is not None:
            with pyro.plate("x", X.shape[0]) : 
                with poutine.scale(scale=1/X.shape[0]):
                    loc, scale=self.Encoder(X[:,self.K:],useK=False)
                    encoded=pyro.sample("z", dist.Normal(loc,scale).to_event(1))
                    _loc, _scale=self.Encoder(X[:,:self.K],useK=True)
                    #D = torch.norm(loc - _loc, p=2)
                   # pyro.factor("D", -D,has_rsample=False)  

                    encodedLong=pyro.sample("zlong", dist.Normal(_loc,_scale).to_event(1))
                    _loc,_scale=self.Encoder_logits((encoded+encodedLong) / 2 )
                    logits_x=pyro.sample('logits_x',dist.Normal(_loc, _scale).to_event(1))
        if X_prime is not None:
            with pyro.plate("xp",X_prime.shape[0]):
                with poutine.scale(scale=1/X_prime.shape[0]):
                    loc_prime, scale_prime=self.Encoder(X_prime)
                    encoded=pyro.sample("z_prime", dist.Normal(loc_prime,scale_prime).to_event(1))
                    _loc,_scale=self.Encoder_logits(encoded)
                    logits_prime=pyro.sample("logits_xprime", dist.Normal(_loc,_scale).to_event(1))

                                    
    def model(self,X=None,X_prime=None,L=None,class_weights=None):
        pyro.module("vae",self)
        if X is not None:
            theta_xK=pyro.param("theta_xK", 2* torch.ones(self.K),constraint=constraints.positive)
            theta_x=pyro.param("theta_x", 2* torch.ones(self.M ),constraint=constraints.positive)
            with pyro.plate("x", X.shape[0]) : 
                with poutine.scale(scale=1/X.shape[0]):
                    loc,scale=X.new_zeros(torch.Size((X.shape[0],self.celltype_fine))), X.new_ones(torch.Size((X.shape[0],self.celltype_fine)))
                    logits_x=pyro.sample('logits_x',dist.Normal(loc, scale).to_event(1))
                    if class_weights is not None:
                        rebalanced_logits = logits_x * class_weights
                    else:
                        rebalanced_logits = logits_x 
                    if L is not None:
                        pyro.sample("L", dist.Categorical(logits=rebalanced_logits).to_event(1), obs=L) 
                    loc,scale=X.new_zeros(torch.Size((X.shape[0],self.latent_dim))), X.new_ones(torch.Size((X.shape[0],self.latent_dim)))
                    encoded=pyro.sample('z',dist.Normal(loc, scale).to_event(1))               
                    gate_logits,nb_logits=self.Decoder(encoded,useK=False)
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits,logits=nb_logits,total_count=theta_x)
                    xhat=pyro.sample('xhatshort',x_dist.to_event(1),obs=X[:,self.K:].type(torch.LongTensor))
                    encoded=pyro.sample('zlong',dist.Normal(loc, scale).to_event(1))             
                    gate_logits,nb_logits=self.Decoder(encoded,useK=True)
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits,logits=nb_logits,total_count=theta_xK)
                    xhat=pyro.sample('xhatlong',x_dist.to_event(1),obs=X[:,:self.K].type(torch.LongTensor))
                                        
        if X_prime is not None:
            theta_prime=pyro.param("theta_x_prime", 2 *  torch.ones(self.M) ,constraint=constraints.positive)
            with pyro.plate("xp", X_prime.shape[0]) : 
                with poutine.scale(scale=1/X_prime.shape[0]):
                    loc,scale=X_prime.new_zeros(torch.Size((X_prime.shape[0],self.celltype_fine))), X_prime.new_ones(torch.Size((X_prime.shape[0],self.celltype_fine)))
                    logits_xprime=pyro.sample('logits_xprime',dist.Normal(loc, scale).to_event(1))
                    loc,scale=X_prime.new_zeros(torch.Size((X_prime.shape[0],self.latent_dim))), X_prime.new_ones(torch.Size((X_prime.shape[0],self.latent_dim)))            
                    encoded=pyro.sample('z_prime', dist.Normal(loc, scale).to_event(1))                
                    gate_logits,nb_logits=self.Decoder(encoded)
                    xprime_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits,logits=nb_logits,total_count=theta_prime)
                    xprime_hat=pyro.sample('xphat',xprime_dist.to_event(1),obs=X_prime.type(torch.LongTensor)) 

class vae(nn.Module):

    def __init__(self, M,hidden_dim_vec, logits):
        """
        Variational Autoencoder (VAE) for cell type inference in spatial transcriptomics.

        Attributes:
            latent_dim (int): Dimension of the latent space.
            celltype_fine (int): Number of cell type logits.

        Parameters:
            M (int): Input dimension representing the number of genes.
            hidden_dim_vec (list of int): Hidden layer dimensions for the encoder and decoder.
            logits (list of int): Dimensions for cell type logits.
        """
        self.latent_dim=hidden_dim_vec[-1]
        self.celltype_fine=logits[-1]
        super().__init__()
        
        #encoding X as latent representation Z
        encoding_layers=[nn.Linear(M,hidden_dim_vec[0]),nn.Dropout(p=0.1), nn.BatchNorm1d(hidden_dim_vec[0]),nn.ReLU()]
        for dim1, dim2 in zip (hidden_dim_vec[:-1], hidden_dim_vec[1:-1] ) : 
            encoding_layers.append(nn.Linear(dim1,dim2))
            encoding_layers.append(nn.Dropout(p=0.1))
            encoding_layers.append(nn.BatchNorm1d( dim2))
            encoding_layers.append(nn.ReLU())  
        encoding_layers.append(nn.Linear(hidden_dim_vec[-2],hidden_dim_vec[-1] * 2))
        encoding_layers.append(nn.Dropout(p=0.1))
        encoding_layers.append(nn.BatchNorm1d( hidden_dim_vec[-1] * 2))
        self.encoder = nn.Sequential(*encoding_layers)
        
        #decoding X_hat 
        decoding_layers=[]
        for dim1, dim2 in zip (hidden_dim_vec[::-1], hidden_dim_vec[::-1][1:] ) : 
            decoding_layers.append(nn.Linear(dim1,dim2))
            decoding_layers.append(nn.Dropout(p=0.1))
            decoding_layers.append(nn.BatchNorm1d( dim2))
            decoding_layers.append(nn.ReLU())        
        decoding_layers.append(nn.Linear(hidden_dim_vec[0],M*2))
        decoding_layers.append(nn.Dropout(p=0.1))
        decoding_layers.append(nn.BatchNorm1d( M*2))
        self.decoder=nn.Sequential(*decoding_layers)
        
        #representing Z as cell type logits
        encoding_logit_layers=[nn.Linear(hidden_dim_vec[-1],logits[0]),nn.Dropout(p=0.1), nn.BatchNorm1d(logits[0]),nn.ReLU()]
        for dim1, dim2 in zip(logits[:-1],logits[1:-1]):
            encoding_logit_layers.append(nn.Linear(dim1,dim2))
            encoding_logit_layers.append(nn.Dropout(p=0.1))
            encoding_logit_layers.append(nn.BatchNorm1d( dim2))
            encoding_logit_layers.append(nn.ReLU())
        encoding_logit_layers.append(nn.Linear(logits[-2],logits[-1]*2))
        encoding_logit_layers.append(nn.Dropout(p=0.1))
        encoding_logit_layers.append(nn.BatchNorm1d(logits[-1]*2))
        self.encoding_logits=nn.Sequential(*encoding_logit_layers)
        
        #decoding cell type logits as Z
        decoding_logits_layers=[]
        for dim1, dim2 in zip(logits[::-1],logits[::-1][1:]):
            decoding_logits_layers.append(nn.Linear(dim1,dim2))
            decoding_logits_layers.append(nn.Dropout(p=0.1))
            decoding_logits_layers.append(nn.BatchNorm1d( dim2))
            decoding_logits_layers.append(nn.ReLU())
        decoding_logits_layers.append(nn.Linear(logits[0],hidden_dim_vec[-1]*2))
        decoding_logits_layers.append(nn.Dropout(p=0.1))
        decoding_logits_layers.append(nn.BatchNorm1d( hidden_dim_vec[-1]*2))
        self.decoding_logits=nn.Sequential(*decoding_logits_layers)
        
    def Encoder(self, X):
        """
        Encoder function for the mean and scale of the latent representation Z.

        Parameters:
            X (torch.Tensor): Input cell by gene matrix.
    
        Returns:
            tuple: Mean and scale of the latent representation.

        """
        encoded=self.encoder(X)
        loc,scale=torch.split(encoded,encoded.shape[1]//2,dim=-1)
        scale=softplus(scale) +epsilon
        return loc, scale
    def Decoder(self, X):
        """
        Decoder function for parameterising X_hat  from the latent representation.
        
        Parameters:
            X (torch.Tensor): Latent representation.
        
        Returns:
            tuple: logits to parameterisze zero inflated negative binomial distr.
        
        """
        decoded=self.decoder(X)
        gate_logits,nb_logits=torch.split(decoded,decoded.shape[1]//2,dim=-1)
        gate_logits=torch.clamp(gate_logits,min=clamp_min,max=clamp_max)
        nb_logits=torch.clamp(nb_logits,min=clamp_min,max=clamp_max)
        return gate_logits, nb_logits
    
    def Encoder_logits(self,X):
        
        encoded=self.encoding_logits(X)
        loc,scale=torch.split(encoded,encoded.shape[1]//2,dim=-1)
        scale=softplus(scale) + epsilon
        return loc, scale
    def Decoder_logits(self,X):
        decoded=self.decoding_logits(X)
        loc,scale=torch.split(decoded,decoded.shape[1]//2,dim=-1)
        scale=softplus(scale) +epsilon
        return loc, scale
    
    #posterior q(z|x)
    def guide (self,X=None,X_prime=None,L=None,class_weights=None):
        pyro.module("vae",self)
        if X is not None:
            with pyro.plate("x", X.shape[0]) : 
                with poutine.scale(scale=1/X.shape[0]):
                    loc, scale=self.Encoder(X)
                    encoded=pyro.sample("z", dist.Normal(loc,scale).to_event(1))
                    loc,scale=self.Encoder_logits(encoded)
                    logits_x=pyro.sample('logits_x',dist.Normal(loc, scale).to_event(1))
        if X_prime is not None:
            with pyro.plate("xp",X_prime.shape[0]):
                with poutine.scale(scale=1/X_prime.shape[0]):
                    loc, scale=self.Encoder(X_prime)
                    encoded=pyro.sample("z_prime", dist.Normal(loc,scale).to_event(1))
                    loc,scale=self.Encoder_logits(encoded)
                    logits_xprime=pyro.sample("logits_xprime", dist.Normal(loc,scale).to_event(1))
    #p(x|z)
    def model(self,X=None,X_prime=None,L=None,class_weights=None):
        pyro.module("vae",self)
        if X is not None:
            theta_x=pyro.param("theta_x", 2* torch.ones(X.shape[1]),constraint=constraints.positive)
            with pyro.plate("x", X.shape[0]) : 
                with poutine.scale(scale=1/X.shape[0]):
                    loc,scale=X.new_zeros(torch.Size((X.shape[0],self.celltype_fine))), X.new_ones(torch.Size((X.shape[0],self.celltype_fine)))
                    logits_x=pyro.sample('logits_x',dist.Normal(loc, scale).to_event(1))
                    if class_weights is not None:
                        rebalanced_logits = logits_x * class_weights
                    else:
                        rebalanced_logits = logits_x 
                    if L is not None:
                        pyro.sample("L", dist.Categorical(logits=rebalanced_logits).to_event(1), obs=L) 
                    loc,scale=X.new_zeros(torch.Size((X.shape[0],self.latent_dim))), X.new_ones(torch.Size((X.shape[0],self.latent_dim)))
                    encoded=pyro.sample('z',dist.Normal(loc, scale).to_event(1))               
                    gate_logits,nb_logits=self.Decoder(encoded)
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits,logits=nb_logits,total_count=theta_x)
                    xhat=pyro.sample('xhat',x_dist.to_event(1),obs=X.type(torch.LongTensor))
        if X_prime is not None:
            theta_prime=pyro.param("theta_x_prime", 2 * torch.ones(X_prime.shape[1]),constraint=constraints.positive)
            with pyro.plate("xp", X_prime.shape[0]) : 
                with poutine.scale(scale=1/X_prime.shape[0]):
                    loc,scale=X_prime.new_zeros(torch.Size((X_prime.shape[0],self.celltype_fine))), X_prime.new_ones(torch.Size((X_prime.shape[0],self.celltype_fine)))
                    logits_xprime=pyro.sample('logits_xprime',dist.Normal(loc, scale).to_event(1))
                    loc,scale=X_prime.new_zeros(torch.Size((X_prime.shape[0],self.latent_dim))), X_prime.new_ones(torch.Size((X_prime.shape[0],self.latent_dim)))            
                    encoded=pyro.sample('z_prime', dist.Normal(loc, scale).to_event(1))                
                    gate_logits,nb_logits=self.Decoder(encoded)
                    xprime_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits,logits=nb_logits,total_count=theta_prime)
                    xprime_hat=pyro.sample('xphat',xprime_dist.to_event(1),obs=X_prime.type(torch.LongTensor)) 
                    
def feature_selection(obj,genes=None,min_perc=0.05,var=0.5):
    '''
    Perform feature selection on genes based on variance and row occurrence.
    Parameters:
        obj (sparrow.read_parquet object): An object created from reading a parquet file with gene data.
        genes (List[str], optional): List of genes to consider. If None, uses genes from the object.
        min_perc (float, optional): Minimum percentage of rows a gene must appear in to be included. Defaults to 0.05.
        var (float, optional): Variance threshold for gene selection. Defaults to 0.5.

    Returns:
        List[str]: List of selected genes after applying variance and row occurrence filters.

    '''
    if genes is None:
        genes=obj._genes
    # Compute variance for each gene
    v=np.var(obj.parquet.loc[:,genes]).compute()
    # Filter genes based on variance threshold
    selected_genes= [gene for gene in np.array(genes)[v>=var]]
    # Further filter genes based on minimum percentage occurrence
    if min_perc > 0:
        gene_counts = obj.parquet.loc[:, selected_genes].astype(bool).sum(axis=0).compute()
        selected_genes = [gene for gene in selected_genes if gene_counts[gene] / len(obj.parquet) > min_perc]
    return selected_genes

def train_test_split(X,X_prime=None,label=None,use_xprime_labels=False,output_path='.',selected_features=None, **kwargs):
    '''
    Split spatial and optional scRNA-seq data and labels into training and testing sets.
    Parameters:
        X (sparrow.read_parquet object): Object containing spatial transcriptomic data.
        X_prime (AnnData, optional): AnnData object of labelled scRNA-seq data of the same tissue origin.
        label (list or str, optional): Labels for the training data. If string, it's used as a column name in X_prime.obs.
        use_xprime_labels (bool, optional): Whether to use labels from X_prime. Defaults to False.
        output_path (str, optional): Path for writing training/testing indices and label encoding dictionary. Defaults to current directory.
        selected_features (list or np.ndarray, optional): Gene features to be used for model training.

    Returns:
        tuple: Training and testing data for spatial and scRNA-seq datasets, and labels if applicable.

        
    '''    

    if selected_features is None:
        genes=feature_selection(X,genes=kwargs.get('genes'), min_perc=kwargs.get('min_perc', 0.01), var=kwargs.get('var', 0.5))
    else:
        genes=selected_features
    # Process scRNA-seq data if provided
    if X_prime is not None:
        assert isinstance(X_prime, AnnData), "X_prime must be an AnnData object."
        genes=list(set(genes) & set(X_prime.var.index))
        np.savetxt(os.path.join(output_path, 'selected_gene_features.txt'), genes, fmt='%s')
        assert len(genes) >= 50, "There are fewer than 50 common genes between ST and scRNA-seq, is the correct scRNA-seq object used?" 
        # Handling labels from scRNA-seq data
        if use_xprime_labels:
            label = X_prime.obs[label].values
        else:
            assert label is not None, "Labels are required when use_xprime_labels is False."
            
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(label)
        split_data = _split_data(X, X_prime, label_encoded, genes, kwargs, output_path)
        
        # Save LabelEncoder dictionary
        _save_label_encoder(label_encoder, output_path)

        return split_data

    # Process ST data only
    else:
        split_data = _split_data(X, None, label, genes, kwargs, output_path)
        
        return split_data

        
def _split_data(X, X_prime, label, genes, kwargs, output_path):
    """
    Internal function to split data into training and testing sets.

    Parameters:
        X (sparrow.read_parquet object or numpy array or torch.Tensor): ST data.
        X_prime (AnnData, optional): scRNA-seq data.
        label (array): Labels for the data.
        genes (list): List of genes to be used.
        kwargs (dict): Additional arguments.
        output_path (str): Path to save output files.

    Returns:
        Tuple: Train and test datasets for ST and scRNA-seq data, along with labels.
    """
    import sklearn.model_selection 
    # Extract ST data
    if isinstance(X, ingest.read_parquet):
        X = X.parquet.loc[:, genes].compute()
    elif isinstance(X, (np.ndarray, torch.Tensor)):
        X = X

    # Extract scRNA-seq data if provided
    if X_prime is not None:
        X_prime_data = np.asarray(X_prime.X[:, np.where(X_prime.var.index.isin(genes))[0]].todense())

    # Split data into training and testing sets
    if label is not None:
        # When labels are provided
        labelencoder = LabelEncoder().fit(label)
        if X_prime is not None:
            train_X_prime, test_X_prime, train_index_prime, test_index_prime, train_label, test_label = sklearn.model_selection .train_test_split(
                X_prime_data, np.arange(len(X_prime_data)), labelencoder.transform(label),
                train_size=kwargs.get('train_perc', 0.7), random_state=kwargs.get('random_state', 42))
            train_X, test_X, train_index, test_index = sklearn.model_selection .train_test_split(
                X, np.arange(len(X)), train_size=kwargs.get('train_perc', 0.7),
                random_state=kwargs.get('random_state', 42))

            # Save indices to disk
            _save_indices(output_path, train_index, test_index, train_index_prime, test_index_prime)
            return (train_X, test_X, train_X_prime, test_X_prime, train_label, test_label)
        else:
            train_X, test_X, train_index, test_index, train_label, test_label = sklearn.model_selection .train_test_split(
                X, np.arange(len(X)), labelencoder.transform(label),
                train_size=kwargs.get('train_perc', 0.7), random_state=kwargs.get('random_state', 42))

            # Save indices to disk
            _save_indices(output_path, train_index, test_index)
            return (train_X, test_X, None, None, train_label, test_label)
    else:
        # When no labels are provided
        train_X, test_X, train_index, test_index = sklearn.model_selection.train_test_split(
            X, np.arange(len(X)), train_size=kwargs.get('train_perc', 0.7), random_state=kwargs.get('random_state', 42))

        # Save indices to disk
        _save_indices(output_path, train_index, test_index)
        return (train_X, test_X, None, None, None, None)

def _save_indices(output_path, train_index, test_index, train_index_prime=None, test_index_prime=None):
    """
    Save training and testing indices to disk.

    Parameters:
        output_path (str): Path to save the indices.
        train_index (array): Indices of training data.
        test_index (array): Indices of testing data.
        train_index_prime (array, optional): Indices of training data for scRNA-seq.
        test_index_prime (array, optional): Indices of testing data for scRNA-seq.
    """
    np.savetxt(os.path.join(output_path, 'spatial_train_index.txt'), train_index, fmt='%d')
    np.savetxt(os.path.join(output_path, 'spatial_test_index.txt'), test_index, fmt='%d')
    if train_index_prime is not None and test_index_prime is not None:
        np.savetxt(os.path.join(output_path, 'scRNAseq_train_index.txt'), train_index_prime, fmt='%d')
        np.savetxt(os.path.join(output_path, 'scRNAseq_test_index.txt'), test_index_prime, fmt='%d')
        
def _save_label_encoder(label_encoder, output_path):
    """
    Save the LabelEncoder dictionary to disk.

    Parameters:
        label_encoder (LabelEncoder): LabelEncoder object.
        output_path (str): Path to save the label encoder dictionary.
    """
    label_encoder_dict = {k: str(v) for k, v in zip(label_encoder.classes_, np.arange(len(label_encoder.classes_)))}
    with open(os.path.join(output_path, 'label_encoder.json'), 'w') as f:
        json.dump(label_encoder_dict, f)

        
def train(model, svi, X,  X_prime=None,label=None,sampling=False,num_epochs=200,log_name='training.log', weighted_training=False, write=False, output_name='vae_model.pkl', **kwargs):
    '''
    Train the VAE model using stochastic variational inference (SVI).

    Parameters:
        model (torch.nn.Module): The VAE model to be trained.
        svi (pyro.infer.SVI): SVI object used for stochastic optimization.
        X (torch.Tensor): Spatial training data array.
        X_prime (torch.Tensor, optional): scRNA-seq training data array. Default: None.
        label (torch.Tensor, optional): Labels for the training data. Default: None.
        sampling(bool, optional): Whether to perform balancedsampling for underrepresented classes. Default:False
        num_epochs (int, optional): Number of training epochs. Default: 200.
        log_name (str, optional): Name of the log file to save training information. Default: 'training.log'.
        write (bool, optional): Whether to write the trained model to a file. Default: False.
        output_name (str, optional): Filename for saving the trained model. Default: 'vae_model.pkl'.
        batch_size (int, optional): Batch size for the DataLoader. Default: 10000.
        shuffle (bool, optional): Whether to shuffle the data. Default: True.

    Returns:
        None

    
    '''
    from torch.utils.data import DataLoader
    # Prepare for training
    model.train()
    pyro.clear_param_store()
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
    # Convert data to PyTorch Tensors if necessary
    X = torch.from_numpy(X).type(torch.float) if isinstance(X, np.ndarray) else X
    X_prime = torch.from_numpy(X_prime).type(torch.float) if isinstance(X_prime, np.ndarray) else X_prime
    label = torch.from_numpy(label).type(torch.LongTensor) if isinstance(label, np.ndarray) else label

    # Handling different data scenarios
    dataset = dataloader.Dataloader(X, X_prime, label)
    print ('loading data')
    if sampling:
        balanced_sampler = dataloader.BalancedSampler(label) if label is not None else None
        data_loader = DataLoader(dataset, sampler=balanced_sampler, batch_size=kwargs.get('batch_size', 10000),shuffle=False)
    else:
        data_loader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 10000),shuffle=True)
    if weighted_training:
        class_counts = torch.bincount(label)
        class_weights = 1. / class_counts.float()
        class_weights = class_weights / class_weights.sum()  # Normalize weights
    else:
        class_weights = None
    # Training loop
    print ('start training')
    for epoch in range(num_epochs):
        total_loss = 0.0       
        for batch in data_loader:
            if len(batch)==3:
                X,X_prime,label=batch
                #loss=svi.step(X=None,X_prime=X,L=None,class_weights=class_weights)
                #loss=svi.step(X=X_prime,X_prime=X,L=None,class_weights=class_weights)
                loss=svi.step(X=X_prime,X_prime=X, L=label,class_weights=class_weights) #for historical reasons, X and X prime here are inverted
            elif len(batch)==2:
                X,label=batch
                loss=svi.step(X=X,X_prime=None,L=None,class_weights=class_weights)
                loss=svi.step(X=X,X_prime=None,L=label,class_weights=class_weights)
            elif len(batch)==1:
                X,=batch
                loss=svi.step(X=X,X_prime=None,L=None,class_weights=class_weights)
            total_loss+=loss
        avg_loss =total_loss/len(data_loader)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    # Save the model if required
    if write:
        torch.save(model.state_dict(),output_name)
        
        
    


    
    
    
    
    