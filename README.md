# Overview of SPARROW
SPARROW is a computational framework that performs integrative cell type inference and microenvironment zone delineation with superior performance compared to state-of-the-art methods and is capable of predicting microenvironment zones for cells lacking spatial context such as those in scRNA-seq. It requires the sole input of an arrow/parquet file containing gene expression and spatial localisation arrays. The manuscript describing the design and advantages of SPARROW can be found here: 
https://www.biorxiv.org/content/10.1101/2024.04.05.588159v1.full.pdf

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
`GAT` delineates neighbourhoods/microenvironment zones by incorporating two sources of information: 1) cell identities encoded as the latent representation Z from the VAE and 2) the spatial proximity patterns between cells. Follow 04_Train_GAT.ipynb notebook for model training and application.
