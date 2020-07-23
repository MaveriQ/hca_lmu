import surprise
import pyarrow.parquet as pq
import argparse
from pathlib import Path
import sys
import pickle
import numpy as np

def args_parser():

    args = argparse.ArgumentParser()

    args.add_argument('--exp_name',type=str,default='test')
    args.add_argument('--num_rows',type=int,default=1000000)
    args.add_argument('--sim_name',type=str,default='pearson')
    args.add_argument('--algorithm',type=str,default='normal')
    args.add_argument('--item_based',action='store_true')
    args.add_argument('--cv_folds',type=int,default=5)

    args = args.parse_args()
    return args

def main(args):

    user_item_based='item_based' if args.item_based else 'user_based'
    filename = '_'.join([args.exp_name,args.algorithm,args.sim_name,user_item_based,str(args.num_rows)])+'.pkl'

    output_file = Path(filename)
    if output_file.exists():
        print(f'ERROR! Output file {output_file} already exists. Exiting!')
        sys.exit(1)

    print(f'Saving scores in {output_file}\n')

    reader = surprise.Reader(rating_scale=(1, 5))
    df = pq.read_table('all_ratings_with_indices.parquet',columns=['user_idx','movie_idx','rating']).to_pandas()
    df.user_idx = df.user_idx.astype(np.uint32)
    df.movie_idx = df.movie_idx.astype(np.uint16)
    df.rating = df.rating.astype(np.uint8)
    print(df.dtypes)
    data = surprise.Dataset.load_from_df(df[:args.num_rows], reader=reader)
    del df
    sim_options = {'name': args.sim_name,
               'user_based':False if args.item_based else True}

    if args.algorithm=='knn':
        algo = surprise.KNNBasic(sim_options=sim_options)
    elif args.algorithm=='baseline':
        algo = surprise.BaselineOnly()
    elif args.algorithm=='normal':
        algo = surprise.NormalPredictor()
    elif args.algorithm=='knn_zscore':
        algo = surprise.KNNWithZScore(sim_options=sim_options)
    elif args.algorithm=='svd':
        algo = surprise.SVD()
    elif args.algorithm=='nmf':
        algo = surprise.NMF()
    else:
        print(f'Algorithm {args.algorithm} is not a valid choice.')

    scores = surprise.model_selection.cross_validate(algo, data, cv=args.cv_folds, verbose=True,n_jobs=-1)

    pickle.dump(scores,open(output_file,'wb'))


if __name__=="__main__":
    args = args_parser()
    main(args)