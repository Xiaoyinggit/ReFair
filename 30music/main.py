# use one pretrain RNN and two mdps
import argparse
import configparser
import torch
import numpy as np
from model import pretrainRNNModel
from userGroup import UserGroup
from mdp import MDP
import data_utils
import pandas as pd
import utils
from ReFair import ReFair
from RL_DM import RL_DM
import json
from torch.optim import Adam


import random
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    parser = argparse.ArgumentParser('pretrain simulator.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_file', type=str, help = "config", default='config/RL_UnFair_config')
    parser.add_argument('--item_feats_file', type=str, help = "config", default='data/30music_2left_new.pkl')

    parser.add_argument('--policy_name', type=str, help = "config", default='greedy')
    parser.add_argument('--log_path', help='logging dir', type=str, default='./log')
    parser.add_argument('--out_dir', help='output dir', type=str, default='./out')


    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))


 
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


    #load model
    train_batch_size = int(config['PRE_RNN']['train_batch_size'])
    item_feats = data_utils.load_item_feats(args.item_feats_file)

    group_num =  int(config['META']['GROUP_NUM'])

    pretrain_RNN_global = pretrainRNNModel(config['PRE_RNN'], item_feats=item_feats, device=device, train_batch_size=train_batch_size)
    pretrain_rnn_model_path = config['PRE_RNN']['model_path']
    print('[main] load global rnn model from %s'%(pretrain_rnn_model_path))

    pretrain_RNN_global.load_state_dict(torch.load(pretrain_rnn_model_path))

    for param in pretrain_RNN_global.parameters():
            param.requires_grad = False

   

    
    T = int(config['META']['T'])
    print('T: ', T)
  
    # load train users
    groups = [UserGroup('Group_%d'%i, config['GROUP%d'%i], batch_size=train_batch_size) for i in range(group_num)]
    for gi in range(group_num):
        g = groups[gi]
        g._init_users(pretrainRNN=pretrain_RNN_global,T=T)
    
    

    mdp_l = [MDP(config['MDP'], T=T, pretrainRNN=pretrain_RNN_global, group_ind=i) for i  in range(group_num)] 

    # initialize policy 
    policy = None
    if args.policy_name == 'RL_PO':
        policy = ReFair(config['PERF_FAIR_POLICY'],groups= groups, phi_sa_normalize= pretrain_RNN_global.phi_sa_normalize, device=device)
    elif args.policy_name == 'RL_DM':
        policy = RL_DM(config['RL_DM'],groups= groups, phi_sa_normalize= pretrain_RNN_global.phi_sa_normalize, device=device)
    elif args.policy_name != 'greedy':
        print('Unknown policy name! %s' %args.policy_name)
        raise AssertionError
    

    

    reward_l = []
    retention_l = []



    for t in range(T):
       
        start_time = time.time()
        for gi in range(group_num):

          if mdp_l[gi].need_calQ(t):
            # need to recalculate Q 
            print('[main]group %d:  re-cal Q at step %d'%(gi,t))
             
            mdp_l[gi].calQ(groups[gi], pretrain_RNN_global, t)

        # do policy optimization
        if (policy is not None) and (policy.needOptimize) and (t>0):
            # update non-argmax 
            c_g_retention = retention_l[-1]
            g_re_rates = [c_g_retention[gi][1]/c_g_retention[gi][2] for gi in range(len(groups)) ]
            g_re_rates_norm = [e/(g_re_rates[1]) for  e in g_re_rates]
            policy.optimize(mdp_l, groups, time=t, g_re_rates=g_re_rates_norm)

        # act accoding to policy, and gather the reward.
        g_reward = [0 for _ in range(group_num)]
        g_retention = [[0,0,0] for _ in range(group_num)] #[# user at time t, #user at time t-1]

        for gi in range(group_num):
           
           tg_reward, tg_retention, _ = mdp_l[gi].act(groups[gi], pretrain_RNN_global, time=t, policy_name=args.policy_name, policy_obj=policy)
           g_reward[gi] = tg_reward
           g_retention[gi] = tg_retention
        
        reward_l.append(g_reward)
        retention_l.append(g_retention)

        torch.cuda.empty_cache()

      

               

       
    # save reward_l / group_retentions.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    utils.save_re(reward_l, retention_l, args.out_dir+'/train_%d'%args.seed, group_num)











if __name__ == '__main__':
    main()

    