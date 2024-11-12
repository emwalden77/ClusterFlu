# ClusterFlu
Employing the Jaccard distance metric to influenza sequences, reducing by PCA, UMAP, and t-SNE, then projecting most recent sequence data onto embedding of previous data.

Plots show:<br> 
-all data after Jaccard is employed then reduced by PCA, UMAP, and t-SNE<br>
-only data until 2019 after Jaccard is employed, reduced by PCA, UMAP, and t-SNE<br>
-only data after 2019 after Jaccard is employed, reduced by PCA, UMAP, and t-SNE<br>-the projection of data after 2019 onto the embedding of data until 2019 found by PCA, UMAP, and t-SNE


## Installation
```bash
pip install scipy scikit-learn openTSNE pandas numpy umap-learn matplotlib seaborn 
```
## Data
Data is collected from NCBI's Influenza Virus Database, selecting for Type A, Human, HA nucleotide sequences from the USA with subtypes H3N2 and H1N1 from 2009 to 2024. 

## Usage
To run the program, go to src to find the Jupyter Notebooks with code for analyzing data for H1N1 and H3N2 

## Results
PCA, UMAP, and t-SNE 2D plots from both H3N2 and H1N1 clustering

<iframe src="/../assets/H1N1_PCA_Date_Transformed_AllData.pdf" width="100%" height="600px"></iframe>


