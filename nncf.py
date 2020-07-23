import torch
import pytorch_lightning as pl
import pyarrow.parquet as pq
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import sys
from select import select
from sklearn.metrics import accuracy_score

class NNCF(pl.LightningModule):

    def __init__(self,args):
        super().__init__()

        self.args = args

        self.movie_embedding = nn.Embedding(args.num_movies,args.embedding_dim)
        self.user_embedding = nn.Embedding(args.num_users,args.embedding_dim)

        self.layer1 = nn.Linear(2*args.embedding_dim,args.hidden_dim)
        self.layer2 = nn.Linear(args.hidden_dim,6) # num_rating+1 to account for ratings 1 to 5

        self.dropout = nn.Dropout(p = args.p_dropout)

    def forward(self, movie_id,user_id):

        m_emb = self.movie_embedding(movie_id)
        u_emb = self.user_embedding(user_id)

        emb = torch.cat([m_emb,u_emb],dim=1)

        out_linear=self.dropout(F.relu(self.layer1(emb)))

        return self.layer2(out_linear)

    def training_step(self, batch, batch_idx):

        user_id,movie_id,rating = batch

        logit = self(movie_id,user_id)

        loss = torch.nn.CrossEntropyLoss()(logit,rating)

        return {'loss':loss, 'log': {'loss':loss}}

    def validation_step(self, batch, batch_idx):

        user_id,movie_id,rating = batch

        logit = self(movie_id,user_id)

        loss = torch.nn.CrossEntropyLoss()(logit,rating)

        y_pred = torch.argmax(logit,axis=1).type(torch.FloatTensor).to(logit.device)
        mse_loss = nn.MSELoss()(y_pred,rating)
        mae_loss = nn.L1Loss()(y_pred,rating)

        return {'val_loss':loss, 'mse':mse_loss, 'mae': mae_loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['mse'] for x in outputs]).mean()
        avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'mse_loss': avg_mse, 'mae_loss': avg_mae}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def prepare_data(self):

        print('Loading Data....')
        df = pq.read_table('all_ratings_with_indices.parquet',columns=['user_idx','movie_idx','rating']).to_pandas()
        user_tensor = torch.LongTensor(df.user_idx.astype(np.uint32))
        movie_tensor = torch.LongTensor(df.movie_idx.astype(np.uint16))
        rating_tensor = torch.LongTensor(df.rating.astype(np.uint8))

        dataset = torch.utils.data.TensorDataset(user_tensor,movie_tensor,rating_tensor)
        len_data = df.shape[0]
        len_val=int(len_data*0.05)
        len_test=int(len_data*0.05)
        len_train=len_data-(len_val+len_test)

        print('Loading Data.... Complete!')
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [len_train,len_val,len_test])

    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(self.train_dataset,pin_memory=True,num_workers=4,batch_size=self.args.batch_size)

        return train_loader

    def val_dataloader(self):

        val_loader = torch.utils.data.DataLoader(self.val_dataset,pin_memory=True,num_workers=4,batch_size=self.args.batch_size)

        return val_loader

    # def test_dataloader(self):

    #     test_loader = torch.utils.data.DataLoader(self.test_dataset,pin_memory=True,num_workers=4,batch_size=self.args.batch_size)

    #     return test_loader


def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_rows',type=int,default=1000000)

    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--p_dropout',type=float,default=0.1)
    parser.add_argument('--hidden_dim',type=int,default=1024)
    parser.add_argument('--embedding_dim',type=int,default=64)
    parser.add_argument('--num_movies',type=int,default=17770)
    parser.add_argument('--num_users',type=int,default=480189)

    parser.add_argument('--exp_name', 
                        type=str,
                        default='default_exp')
    parser.add_argument('--tags', 
                        nargs='*',
                        type=str,
                        default=[])
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10)
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1)
    parser.add_argument('--val_check_interval',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_val_sanity',
                        type=int,
                        default=1)
    parser.add_argument('--limit_val_batches',
                        type=int,
                        default=1)
    parser.add_argument('--gpus', 
                        type=int,
                        # nargs='*',
                        default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--accum_batches',
                        type=int,
                        default=1)
    parser.add_argument('--weights_summary', 
                        type=str,
                        default=None)
    parser.add_argument('--grad_clip_val',
                        type=float, 
                        required=False,
                        default=0.0)
    parser.add_argument('--fp16',
                        action='store_true', 
                        required=False)
    parser.add_argument('--profile',
                        action='store_true', 
                        required=False)
    parser.add_argument('--dev_run',
                        action='store_true', 
                        required=False)
    parser.add_argument('--track_grads',
                        action='store_true', 
                        required=False)
    parser.add_argument('--auto_lr',
                        action='store_true', 
                        required=False)
    parser.add_argument('--seed',
                        type=int,
                        default=42)

    args = parser.parse_args()

    if args.dev_run:
        args.tags.append('dev_run')

    if args.profile:
        args.tags.append('profile')

    if args.fp16:
        args.tags.append('fp16')
    else:
        args.tags.append('fp32')

    return args

def print_args(args):
    args_dict = vars(args)
    print('\n')
    print('*'*20+' Experiment Parameters '+'*'*20+'\n')
    for k,v in (args_dict.items()):
        print('\t{0: <20}:\t{1}'.format(k,v))
    print('\n'+'*'*65+'\n')
    print('Press ENTER key to confirm the parameters or wait 30 seconds to continue automatically ...')
    timeout = 30
    rlist, wlist, xlist = select([sys.stdin], [], [], timeout)

    if rlist:
        print('Thank you. Continuing.. ')
    else:
        print('No key pressed. Continuing.. ')

def main(args):
    seed_everything(args.seed)
    model = NNCF(args)

    checkpoint_callback = ModelCheckpoint(
                                            filepath='./checkpoints/nncf_{step}-{val_loss:.3f}',
                                            save_top_k=-1,
                                            verbose=True,
                                            monitor='val_acc',
                                            mode='max',
                                            prefix='',
                                            period=1,
                                        )

    tb_logger = TensorBoardLogger(
                                    save_dir=os.getcwd(),
                                    version=1,
                                    name='lightning_logs'
                                    )
                                    
    trainer = pl.Trainer(fast_dev_run=True if args.dev_run else False,
                    weights_summary=args.weights_summary,
                    num_sanity_val_steps=args.num_val_sanity,
                    gpus=args.gpus,
                    distributed_backend='dp',
                    benchmark=True,
                    amp_level='O1', 
                    precision=16 if args.fp16 else 32,
                    deterministic=False,
                    accumulate_grad_batches=args.accum_batches,
                    auto_lr_find=True if args.auto_lr else False,
                    checkpoint_callback=checkpoint_callback,
                    # early_stop_callback=early_stop,
                    # callbacks=callbacks,
                    gradient_clip_val=args.grad_clip_val,
                    limit_val_batches=args.limit_val_batches,
                    # max_steps=args.num_steps,
                    max_epochs=args.num_epochs,
                    val_check_interval=args.val_check_interval,
                    profiler=AdvancedProfiler(output_filename='profile_report.txt') if args.profile else None,
                    track_grad_norm=2 if args.track_grads else -1,
                    logger=tb_logger)

    trainer.fit(model)
    
if __name__=='__main__':

    args = args_parser()
    print_args(args)
    main(args)