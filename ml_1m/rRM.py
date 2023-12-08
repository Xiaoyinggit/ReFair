import argparse
import configparser
import torch
import numpy as np
from model import pretrainRNNModel
from userGroup import UserGroup
from ReFair import ReFair
from mdp import MDP
import data_utils
import pandas as pd
import utils
import json
from torch.optim import Adam


import random
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class dRO:

    def __init__(self, rrm_config, pretrainRNN, device):
        super(dRO, self).__init__()

        self.rrm_config = rrm_config
        print('[dRO] dRO config: ', dict(self.rrm_config))

        self.device = device
        self.dim = int(self.rrm_config['dim'])
        self.lr = float(self.rrm_config['lr'])
        self.lamda= float(self.rrm_config['lambda'])
        self.eta = float(self.rrm_config['eta'])
        self.item_count  = pretrainRNN.item_count
        self.itemEmb=pretrainRNN.item_tower(torch.arange(start=0, end=self.item_count, device=self.device))
        self.normalize = pretrainRNN.phi_sa_normalize



        self.w = torch.nn.Linear(self.dim, 1, device=self.device,bias=False, dtype=torch.float64)  # [d,1] for calculate Q
        self.gw = 0.5 * torch.ones(2, dtype=torch.float64, device=self.device, requires_grad=False) #group weight


        self.optimizer = Adam(self.w.parameters(), lr=self.lr)




    def getPred(self, u_states, action):
        
        actionEmb = self.itemEmb[action,:] #[vU, d]
        phi_sa = torch.mul(u_states, actionEmb) #[vU, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 


        r_hat = self.w(phi_sa).squeeze(dim=-1)

        return r_hat
    
    def update(self, interaction_l):
          
        
        
        loss_vec_l, loss_mean_l = [], []
        for in_t in interaction_l:
            phi_sa, action, r_label = in_t
            r_label = r_label.squeeze(dim=-1)
            r_hat = self.getPred(phi_sa, action)
            #print('[rrm] update: ', r_hat.size(), r_label.size())

            g_loss = torch.square(r_hat-r_label)

            loss_vec_l.append(g_loss)
            loss_mean_l.append(torch.mean(g_loss.detach()).unsqueeze(dim=-1))

        # update w 
        g_l_mean = torch.cat(loss_mean_l, dim=-1)
        exp_l = torch.exp(self.eta * g_l_mean)
        self.gw = torch.mul(self.gw, exp_l)
        self.gw = self.gw/ torch.sum(self.gw)

        loss_vec = torch.cat([self.gw[0]* loss_vec_l[0], self.gw[1]* loss_vec_l[1]], dim=0)

        
        loss = loss_vec.mean() + self.lamda * self.w.weight.norm()
        print('loss: ', loss_vec.mean().item(), self.lamda * self.w.weight.norm().item(), loss.item(), self.gw.tolist(), g_l_mean.tolist())
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def excute(self, u_states, u_selected_item):
        
        phi_sa = torch.mul(u_states.unsqueeze(dim=1), self.itemEmb.unsqueeze(dim=0))  #  #[vU,1, d] \times [1,A, d] --> [vU, A, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 
        
        r_hat = self.w(phi_sa).squeeze(dim=-1) #[vU,A]

        r_hat = r_hat - u_selected_item

        action = torch.argmax(r_hat, dim=-1, keepdim=False)

        return action


class rRM_fair:

    def __init__(self, rrm_config, pretrainRNN, device):
        super(rRM_fair, self).__init__()

        self.rrm_config = rrm_config
        print('[rRM_fair] rRM_fair config: ', dict(self.rrm_config))

        self.device = device
        self.dim = int(self.rrm_config['dim'])
        self.lr = float(self.rrm_config['lr'])
        self.lamda= float(self.rrm_config['lambda'])
        self.fair_lambda =  float(self.rrm_config['fair_lambda'])
        self.item_count  = pretrainRNN.item_count
        self.itemEmb=pretrainRNN.item_tower(torch.arange(start=0, end=self.item_count, device=self.device))
        self.normalize = pretrainRNN.phi_sa_normalize
        


        self.w = torch.nn.Linear(self.dim, 1, device=self.device,bias=False, dtype=torch.float64)  # [d,1] for calculate Q


        self.optimizer = Adam(self.w.parameters(), lr=self.lr)





    def getPred(self, u_states, action):
        
        actionEmb = self.itemEmb[action,:] #[vU, d]
        phi_sa = torch.mul(u_states, actionEmb) #[vU, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 


        r_hat = self.w(phi_sa).squeeze(dim=-1)

        return r_hat
    
    def excute(self, u_states, u_selected_item):
        
        phi_sa = torch.mul(u_states.unsqueeze(dim=1), self.itemEmb.unsqueeze(dim=0))  #  #[vU,1, d] \times [1,A, d] --> [vU, A, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 
        
        r_hat = self.w(phi_sa).squeeze(dim=-1) #[vU,A]

        r_hat = r_hat - u_selected_item

        action = torch.argmax(r_hat, dim=-1, keepdim=False)

        return action
    
    def update(self, interaction_l):
        
        loss_l = []
        for in_t in interaction_l:
            phi_sa, action, r_label = in_t
            r_label = r_label.squeeze(dim=-1)
            r_hat = self.getPred(phi_sa, action)

            g_loss = torch.square(r_hat-r_label)


            loss_l.append(g_loss)
        

        loss_vec = torch.cat(loss_l, dim=0)
        fair_vio = loss_l[0].mean() - loss_l[1].mean()
        fair_loss = torch.square(fair_vio)
        
        loss = loss_vec.mean() + self.lamda * self.w.weight.norm() + self.fair_lambda* fair_loss
        print('--loss: ', loss_vec.size(), loss_vec.mean().item(), self.lamda * self.w.weight.norm().item(), loss.item(), 'fair_loss: ', [fair_vio.item(),fair_loss.item()], 'fair_lambda:', self.fair_lambda)

        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()
       
       


class rRM:

    def __init__(self, rrm_config, pretrainRNN, device):
        super(rRM, self).__init__()

        self.rrm_config = rrm_config
        print('[rRM] rRM config: ', dict(self.rrm_config))

        self.device = device
        self.dim = int(self.rrm_config['dim'])
        self.lr = float(self.rrm_config['lr'])
        self.lamda= float(self.rrm_config['lambda'])
        self.item_count  = pretrainRNN.item_count
        self.itemEmb=pretrainRNN.item_tower(torch.arange(start=0, end=self.item_count, device=self.device))
        self.normalize = pretrainRNN.phi_sa_normalize
        


        self.w = torch.nn.Linear(self.dim, 1, device=self.device,bias=False, dtype=torch.float64)  # [d,1] for calculate Q


        self.optimizer = Adam(self.w.parameters(), lr=self.lr)





    def getPred(self, u_states, action):
        
        actionEmb = self.itemEmb[action,:] #[vU, d]
        phi_sa = torch.mul(u_states, actionEmb) #[vU, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 


        r_hat = self.w(phi_sa).squeeze(dim=-1)

        return r_hat
    
    def excute(self, u_states, u_selected_item):
        
        phi_sa = torch.mul(u_states.unsqueeze(dim=1), self.itemEmb.unsqueeze(dim=0))  #  #[vU,1, d] \times [1,A, d] --> [vU, A, d]

        if self.normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 
        
        r_hat = self.w(phi_sa).squeeze(dim=-1) #[vU,A]

        r_hat = r_hat - u_selected_item

        action = torch.argmax(r_hat, dim=-1, keepdim=False)

        return action
    
    def update(self, interaction_l):
        
        loss_l = []
        for in_t in interaction_l:
            phi_sa, action, r_label = in_t
            r_label = r_label.squeeze(dim=-1)
            r_hat = self.getPred(phi_sa, action)

            g_loss = torch.square(r_hat-r_label)


            loss_l.append(g_loss)
        

        loss_vec = torch.cat(loss_l, dim=0)
        
        loss = loss_vec.mean() + self.lamda * self.w.weight.norm()
        print('--loss: ', loss_vec.size(), loss_vec.mean().item(), self.lamda * self.w.weight.norm().item(), loss.item())

        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()
       

        
      


def main():
    parser = argparse.ArgumentParser('pretrain simulator.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_file', type=str, help = "config", default='config/rrm_config')
    parser.add_argument('--item_feats_file', type=str, help = "config", default='data/ml_1m_diff2week.pkl')

    parser.add_argument('--policy_name', type=str, help = "config", default='greedy')
    parser.add_argument('--rnn_model_path',type=str, default='./ml_1m_rnn_models/ml_1m_pretrained_rnn_ckpt')
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
    pretrain_RNN = pretrainRNNModel(config['PRE_RNN'], item_feats=item_feats, device=device, train_batch_size=train_batch_size)
    pretrain_RNN.load_state_dict(torch.load(args.rnn_model_path))
    # frozen parameters
   
    for param in pretrain_RNN.parameters():
        param.requires_grad = False
    
    T = int(config['META']['T'])
    greedy_T = int(config['META']['greedy_t'])
    print('T: ', T, 'greedy_T', greedy_T)

    group_num =  int(config['META']['GROUP_NUM'])
    # load train users
    groups = [UserGroup('Group_%d'%i, config['GROUP%d'%i], batch_size=train_batch_size) for i in range(group_num)]
    for g in groups:
        g._init_users(pretrainRNN=pretrain_RNN,T=T)


    mdp_l = [MDP(config['MDP'], T=T, pretrainRNN=pretrain_RNN, group_ind=i) for i  in range(group_num)] 

    mdp_policy = ReFair(config['PERF_FAIR_POLICY'],groups= groups, phi_sa_normalize= pretrain_RNN.phi_sa_normalize, device=device)




    reward_l = []
    retention_l = []
    interaction_l = []

    # 0-startT, use greedy max_Q
    for t in range(greedy_T):

        for gi in range(group_num):

          if mdp_l[gi].need_calQ(t):
            # need to recalculate Q 
            print('[main]group %d:  re-cal Q at step %d'%(gi,t))
             
            mdp_l[gi].calQ(groups[gi], pretrain_RNN, t)


        # do policy optimization
        if (mdp_policy is not None) and (mdp_policy.needOptimize) and (t>0):
            # update non-argmax 
            c_g_retention = retention_l[-1]
            g_re_rates = [c_g_retention[gi][1]/c_g_retention[gi][2] for gi in range(len(groups)) ]
            g_re_rates_norm = [e/(g_re_rates[1]) for  e in g_re_rates]
            mdp_policy.optimize(mdp_l, groups, time=t, g_re_rates = g_re_rates_norm)

         # act accoding to policy, and gather the reward.
        g_reward = [0 for _ in range(group_num)]
        g_retention = [[0,0,0] for _ in range(group_num)] #[# user at time t, #user at time t-1]

        debug_ar_l = []
        for gi in range(group_num):
           
           tg_reward, tg_retention, g_interactions = mdp_l[gi].act(groups[gi], pretrain_RNN, time=t, policy_name='perf_fair', policy_obj=mdp_policy)
           g_reward[gi] = tg_reward
           g_retention[gi] = tg_retention
           debug_ar_l.append(g_interactions)
        

       

        reward_l.append(g_reward)
        retention_l.append(g_retention)
        interaction_l.append(debug_ar_l)

        


    # then do rRM
    #linear model
    policy = None
    if args.policy_name =='rrm':
       policy = rRM(config['RRM'], pretrain_RNN, device)
    elif args.policy_name == 'dro':
       policy = dRO(config['DRO'], pretrain_RNN, device)
    elif args.policy_name == 'rrm_fair':
       policy = rRM_fair(config['RRM_FAIR'], pretrain_RNN, device)
    else:
        raise AssertionError

    # init rrm with interactions_l
    for ti in range(len(interaction_l)):
       interaction_t = interaction_l[ti]
       policy.update(interaction_t)


    for t in range(greedy_T, T):
        print('t: ', t)
        
        t_g_reward = [0 for _ in range(len(groups))]
        t_g_retention = [[0,0,0] for _ in range(len(groups))] #[# user at time t, #user at time t-1]
        t_interaction = []
        
        for gi in range(len(groups)):
            g = groups[gi]
            u_states,  u_featEmb, u_rnnSt, u_sItem, appear_uNum = g.get_validUsers(t, return_UN=True) #[vU, d]
            u_active_mask = g.active_user_mask
            #print('[act] user_num %d in group %s'% (u_states.size()[0], g.group_name))

            t_g_retention[gi][0] = u_states.size()[0]
            t_g_retention[gi][2] = appear_uNum.item()

            action_t = policy.excute(u_states, u_selected_item=u_sItem)

            reward, next_state = mdp_l[gi].real_execute(u_states, action_t, pretrain_RNN)

            t_g_reward[gi] = np.power(mdp_l[gi].gamma, t) * torch.sum(reward).item()


            t_interaction.append([u_states, action_t, torch.multiply(reward, 1-(next_state==mdp_l[gi].leave_state).to(torch.float64))])



            # updates user states, according to the real excution.
            leave_mask = 1-(next_state==mdp_l[gi].leave_state).to(torch.float64) # 1 for stay, 0 for leave
            t_g_retention[gi][1] = torch.sum(leave_mask).item()
            

            sample_interactions = torch.concat([action_t.unsqueeze(-1),(reward >= mdp_l[gi].reward_thre).to(int)], dim=-1) #[vU,2] 
            next_u_states, next_u_rnnSt = utils.get_next_userState(u_featEmb, pretrain_RNN, interactions=sample_interactions, user_histSeq=u_rnnSt)
            # set next_u_states & next_u_rnnSt of leave users as zero vector
         
            next_u_states = torch.where(leave_mask>0, next_u_states, torch.zeros_like(next_u_states))
            next_u_rnnSt = torch.where(leave_mask>0, next_u_rnnSt, torch.zeros_like(next_u_rnnSt))

            g.update(next_u_states, next_u_rnnSt, leave_mask, action_t)

            
        
        reward_l.append(t_g_reward)
        retention_l.append(t_g_retention)

        # update Model

       
        policy.update(t_interaction)

    
        
    # save reward_l / group_retentions.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    
    
    utils.save_re(reward_l, retention_l, args.out_dir+'/train_{}'.format(args.seed), group_num)


           
  
if __name__ == '__main__':
    main()





      

    


     
        
    