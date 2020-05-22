# hca_lmu
HCA project at LMU

## Dataset
Download the netflix dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data/download
After extracting the zip file, you will have access to the files used in the notebooks.

## Software Requirements.
Please install following python packages
- pandas
- pyspark (If using pip, install as 'pip install pyspark[sql]' to include SQL bindings)

### Part 1- Data Cleaning - Movie Titles
This notebook deals with cleaning up the movie_titles.csv that we get from the zip archive.
I have not removed the likely errors, and instead showed how to deal with errors. 
The notebook is heavily commented to show likely steps someone would take in cleaning up such a file. 

### Part 2 - Data Cleaning - Ratings
This notebook extracts and cleans up the rating files. Since they are four identical files, I have given detailed steps for one of them and in the end repeated the steps for rest three. 
Here again, I have adopted a tutorial-like approach to help the reader understand a file cleaning pipeline

### Part 3 - PySpark-Recommendation
This notebook gives a step by step tutorial to run a recommendation system on PySpark. 
