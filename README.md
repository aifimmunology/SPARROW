# Overview of SPARROW
SPARROW is a computational framework that performs integrative cell type inference and microenvironment zone delineation with superior performance compared to state-of-the-art methods and is capable of predicting microenvironment zones for cells lacking spatial context such as those in scRNA-seq. It requires the sole input of an arrow/parquet file containing gene expression and spatial localisation arrays. The manuscript describing the design and adantages of SPARROW can be found here: 

Below is the overarching logic flow of this repo.
![alt text](https://github.com/peiyaozhao617/SPARROW/blob/main/doc/architecture.png)

## Dependency
See requirements.txt for dependency.

## Quickstart
### Install and setup
Follow 01_Install_and_Setup.ipynb notebook for setting up the proper environment for SPARROW.
### ingest data, make parquet file and write metrics
Follow 02_Data_Ingest_to_Make_Parquet.ipynb notebook for data ingestion steps. Examples on merFISH (Vizgen) and Xenium ingestion were given.
In brief, the required files are
1. a cell x gene matrix containing transcript counts of cells
2. a cell metadata file containing the bounding spatial coordinates for each cell, named 'x_min','x_max','y_min' and 'y_max', and the centroids of cells, named 'x' and 'y'. 

Additionally, this step also performs some basic data cleanup procedures, for instance, the removal of cells with exceptionally low transcript levels, as these are likely to represent noise in the data. 

### use SPARROW-VAE for cell typing
The `vae` module within SPARROW facilitates cell type inference, in both supervised and unsupervised modes. Follow 03_Train_VAE.ipynb notebook to run SPARROW-VAE in the presence or absence of a cell type labelled orthogonal scRNA-seq dataset.

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
