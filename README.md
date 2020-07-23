# hca_lmu
HCA project at LMU

## Dataset
Download the netflix dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data/download
After extracting the zip file, you will have access to the files used in the notebooks.

## Software Requirements.
Please install following python packages
- pandas
- pyspark (If using pip, install as 'pip install pyspark[sql]' to include SQL bindings)
- pytorch
- surprise
- scipy
- scikit-learn
- pyarrow
- tqdm
- pickle
- numpy
- argparse
- select
- pytorch_lightning

All of these packages can be install using pip. 

### Part 1- Data Cleaning - Movie Titles
This notebook deals with cleaning up the movie_titles.csv that we get from the zip archive.
I have not removed the likely errors, and instead showed how to deal with errors. 
The notebook is heavily commented to show likely steps someone would take in cleaning up such a file. 

### Part 2 - Data Cleaning - Ratings
This notebook extracts and cleans up the rating files. Since they are four identical files, I have given detailed steps for one of them and in the end repeated the steps for rest three. 
Here again, I have adopted a tutorial-like approach to help the reader understand a file cleaning pipeline

### Part 3 - PySpark-Recommendation
This notebook gives a step by step tutorial to run a recommendation system on PySpark. 

### Part 4 - sklearn Recommendation
This notebook gives a description of how to run KNN on the dataset. I also talked about the problems that arose from big dataset and memory constraints. 

### Part 5 - Combination of Algorithms (multiple_alg.py)
We used surprise package to automate the steps in Part 4 and used various similarity measures and algorithm implementations. Essentially we used 
- Cosine Similarity
- Mean Square Distance
- Pearson Correlation

for similarity measures and 
- Singular Value Decompostion
- Non negative Matrix Factorization

for modeling the user-item matrix. We could have used either ALS (Alternating Least Squares) or gradient descent as optimization mechanism, but we chose ALS as that is more commonly used in Recommendation Systems

To run the file, please install surprise package by 'pip install surprise'. The script has a help file of its own

### Part 6 - Neural Network Implementation
We used a simple feed forward neural network with embeddings for a proof of concept (PoC) implementation. The file nncf.py has a built-in help that can be accessed using 'python nncf.py -h'.
This will enable the user to select various parameters for the neural network. 

The results will be logged in a tensorboard. To see the results, run 'tensorboard --log-dir lightning_logs/' and then open the link shown in the output. It will show the loss/metric curves as training progresses. 

## Contact
For any query, please feel free to contact haris.jabbar@gmail.com or raise a github issue.
