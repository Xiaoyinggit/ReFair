import argparse
import configparser
import torch
import numpy as np
from model import pretrainRNNModel
from mdp import MDP
import data_utils
import pandas as pd
from torch.utils.data import DataLoader
import utils
import pickle

import random
import time


def main():
    parser = argparse.ArgumentParser('pretrain simulator.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_file', type=str, help = "config", default='config/rnn_config')

    parser.add_argument('--data_path', type=str, help = "config", default='data/ml_1m.pkl')
    parser.add_argument('--rnn_model_path',type=str, default='data/ml_1m_rnn/epoch_0')
    parser.add_argument('--out_dir', help='output dir', type=str, default='./out')

    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))
    train_batch_size = int(config['META']['train_batch_size'])



    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    data = data_utils.RNNdata(args.data_path, int(config['META']['max_seq_len']), d_type = 'all')
    data_loader = DataLoader(data, batch_size=train_batch_size, shuffle=True)

    user_feats = torch.tensor(np.array(data.data_dict['user_feat'], dtype=np.int64), dtype=torch.int64, device=device) #[gU, 5]


    #load model
    pretrain_RNN = pretrainRNNModel(config['MODEL'], item_feats=data.data_dict['item_feats'], device=device, train_batch_size=train_batch_size)
    pretrain_RNN.load_state_dict(torch.load(args.rnn_model_path))
    print('[load model] Finishing loading models from ', args.rnn_model_path)
    # frozen parameters
   
    pretrain_RNN.eval()

    user_featEmb = pretrain_RNN.get_user_featEmb(user_feats).tolist()

    with open('data/ml_1m_diff2week/user_featEmb.pkl', 'wb') as f:
         pickle.dump(user_featEmb, f, pickle.HIGHEST_PROTOCOL)





   

    pred_re = []
    for batch_samples in data_loader:
        padded_samples, valid_bs = utils.do_padding(batch_samples, device=device, batch_size=train_batch_size)
        r_pred, ns_pred= pretrain_RNN(padded_samples, valid_bs) # [rating_pred, ns_pred]

        ns_pred = torch.nn.functional.softmax(ns_pred, dim=-1)


        user_feats, item_IDs, label, hist_seq, seq_len, ns = batch_samples 

        hist_posn_l = []

        for hi in range(hist_seq.size()[0]):
            hist_fb_l = hist_seq[hi, 0:seq_len[hi],1]
            pos_fb_n = hist_fb_l.sum().item()
            hist_posn_l.append(pos_fb_n)


        s_pred = []
        
        for i in range(user_feats.shape[0]):
            uf = user_feats[i,:].tolist()
            uf.append(seq_len[i].item())
            ns_pred_l = ns_pred[i,:].tolist()
            s_pred = [*uf, item_IDs[i].item(), ns[i].item() , *ns_pred_l, r_pred[i].item(), label[i].item(), hist_posn_l[i]]
            pred_re.append(s_pred)
  

    pred_df = pd.DataFrame(pred_re, columns=['user_id', 'u_gender', 'u_age', 'u_occup', 'u_zip', 'u_seq_len', 'item_id', 'ns', 'ns_pred_0', 'ns_pred_1', 'r_pred', 'r_label', 'hist_posfb_n'])
    pred_df.to_csv('./data/ml_1m_diff2week/ml_1m_pred.csv')




if __name__  == '__main__':
    main()