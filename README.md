# Overview of SPARROW
SPARROW is a two part sequential algorithm for single cell resolution spatial transcriptomics data, requiring the sole input of an arrow/parquet file containing gene expression and spatial localisation arrays. It infers cell types and identifies microenvironment zones representing distinct cell neighborhood types in a unified framework, with superior performance compared to state-of-the-art methods.

Below is the overarching logic flow of this repo.
![alt text](https://github.com/peiyaozhao617/SPARROW/blob/main/doc/architecture.png)

## Dependency
See requirements.txt for dependency.

## Quickstart
### ingest data, make parquet file and write metrics
The `preprocessing` step is designed to ingest output files from various spatial transcriptomics platforms. The required files are 
1. a cell x gene matrix containing transcript counts of cells
2. a cell metadata file containing the bounding spatial coordinates for each cell, named 'x_min','x_max','y_min' and 'y_max', and the centroids of cells, named 'x' and 'y'. 
`preprocessing` can also accept an anndata file, in which the `obs` dataframe contains columns labelled `cell`, `x` and `y`, which are cell names and their corresponding spatial centroid coordinates. Additionally, this step also performs some basic data regulization procedures, for instance, the removal of cells with exceptionally low transcript levels, as these are likely to represent noise in the data. 

To load and run as a module, follow the steps below:
```python
from SPARROW import preprocessing
#Define the path to your cell-by-gene matrix and cell metadata files
cell_by_gene_path = '/path/to/cell_by_gene_matrix.csv'
cell_metadata_path = '/path/to/cell_metadata.csv'
# Alternatively, for anndata format:
anndata_path = '/path/to/your_anndata_file.h5ad'

#accepted formats are {'Csv','Anndata'}
# Initiate the preprocessing routine
# For CSV format:
obj=preprocessing.ingest.make_parquet(cell_by_gene=cell_by_gene_path,cell_meta=cell_metadata_path)  
#if Anndata
obj=preprocessing.ingest.make_parquet(cell_by_gene=anndata_path,input_format='Anndata')
#filter out bad cells containing low numbers of transcripts, which likely represent noise in the data. If no filtering is needed, set threshold=0
obj.filt(lower_threshold=0,upper_threshold=300,output_name='/path/to/parquet',output_fmt='parquet',output_name_prefix='prefix_to_parquet')
#write metrics to a file named metric.txt
obj.write_metric(output_name='metric.txt')
```
### use SPARROW-VAE for cell typing
The `vae` module within SPARROW facilitates cell type inference, in both supervised and unsupervised modes.
#### Example 1: Supervised mode with labelled scRNA-seq data
When labelled scRNA-seq of the same tissue origin is available, cells in the scRNA-seq data are co-embedded with cells in the spatial transcriptomics data. Below are the steps to use the `vae` module for co-embedding and cell type inference, assuming the availability of an annotated scRNA-seq AnnData file (here referred to as X_prime.h5ad).
```python
import scanpy as sc
#load the scRNA-seq anndata
scRNA_seq=sc.read_h5ad('/path/to/X_prime.h5ad')
from SPARROW import preprocessing
#load the previously created parquet file
obj=preprocessing.ingest.read_parquet('/path/to/parquet',n_meta_col=8)
#Split data into training and testing sets for spatial and scRNA-seq data
from SPARROW import VAE
X_train,X_test,X_prime_train,X_prime_test,label_train,label_test=VAE.vae.train_test_split(sparrow_obj,scRNA_seq,label='celltypes',use_xprime_labels=True,train_perc=0.7,random_seed=42)
#Initialize VAE model with specific dimensions for input ( X_train.shape[1]) ,hidden layers ([200,100,80]), and logit layers ([50,40,14]). The last dimension of logits layers need to be the same as the number of cell types defined in accompanying scRNA-seq data.
VAEmodel=VAE.vae.vae( X_train.shape[1],[200,100,80],[50,40,14])
# Set up the optimizer for training
import pyro.optim
from pyro.infer import SVI, config_enumerate, Trace_ELBO
AdamArgs={ 'lr': 1e-3,'weight_decay':1e-5}
optimizer=pyro.optim.Adam(AdamArgs)
svi=SVI(VAEmodel.model,  config_enumerate(VAEmodel.guide, "sequential",expand=True), optimizer, loss=Trace_ELBO())
#Begin training and monitor the loss. The loss is written to a log file with a user defined name.
VAE.vae.train(VAEmodel,svi,X_train,X_prime_train,label_train,num_epochs=200,log_name='training.log') 
# Save the trained model once training stabilizes
VAE.vae.write(VAEmodel,'output.name')
```
Example 2 when labelled scRNA-seq of the same tissue origin is *not* available:
```python
#read the parquet file from preprocessing.ingest.make_parquet
from SPARROW import preprocessing
obj=preprocessing.ingest.read_parquet('/path/to/parquet',n_meta_col=8)
from SPARROW import VAE
X_train,X_test,_,_,_,_=VAE.vae.train_test_split(obj,train_perc=0.7,random_seed=42,min_perc=0,var=0)

VAEmodel=VAE.vae.vae( X_train.shape[1],[100,80,40],[0,0,0])

```
### use SPARROW-GAT for microenvironment zone delineation
`GAT` delineates neighbourhoods/microenvironment zones by incorporating two sources of information: 1) cell identities encoded as the latent representation Z from the VAE and 2) the spatial proximity patterns between cells.
:
```python
#start GAT model by defining the dimensions of layers. In the example below, 80 is the input dimension, which needs to be the same as that of the VAE latent representation Z. 40 is the latent dimension. 5 is the output dimension, which is the number of microenvironment zones that the user defines based on prior knowledge and expectation. The user can test serveral different values for optimal delineation. 3 is the number of attention heads in the graph attention network and 2 is the number of attention network layers in the architecture. To prevent oversmoothing, keep the number of layers relatively low.

from SPARROW import GAT
import torch.optim
model_spatial=GAT.neighbourhoods.minCUTPooling(80,40,5,3,2)
optimizer_gat = torch.optim.Adam(model_spatial.parameters(), lr=5e-3,weight_decay=1e-5)

#load the spatial neighbourhood information as a list of adjacency matrices and gene expression of these cells as a list of cell by gene matrices. The obj in the example below is the SPARROW read_parquet object. Genes are the selected feature genes used in SPARROW VAE. In the example below, 0,20000,0,20000 are spatial coordinates x0,x1,y1,y2 of the bounding box outlining the spatial area to be used for training. 5000 is the step size for tiling the area of interest. Make sure the step size is large enough such that within each tile the types of neighbourhoods are adequately represented. 7 is the neighbourhood size for spatial weight matrix calculation and 5 is the minimum number of transcripts each cell needs to express in order to be represented in the graph.

A_list, X_list=GAT.preprocessing.graph_loader(obj, genes,0,20000,0,20000,5000,7,5) 
#start training
GAT.neighbourhoods.train(model_spatial,VAEmodel, optimizer_gat,A_list,X_list,num_epochs=200,log_name='GATtraining.log')
```
