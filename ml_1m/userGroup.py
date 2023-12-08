import torch
import numpy as np
import pickle
import utils




class UserGroup():

    def __init__(self, group_name, group_config, batch_size, is_train=True, device='cuda'):
        self.group_name = group_name
        self.group_config = group_config
        self.device = device
        self.batch_size = batch_size
        self.appeared_users_num = None
        self.is_train = is_train

    
    def get_activeUT(self, targets):
        # only get actives user state
       non_zero = torch.nonzero(self.active_user_mask).squeeze(-1)

       re = []
       for t in targets:
          re.append(t[non_zero,:])
       return re, self.active_user_mask


    def get_validUsers(self, time, return_UN=False):
       
       # activate users at time $t$.
       if time >0:
           new_users = (self.user_time_slot == time)
           self.active_user_mask = self.active_user_mask + new_users

       if return_UN:
          self.appeared_users_num = torch.sum(self.user_time_slot<=time)

       active_re, _ = self.get_activeUT([self.user_states, self.rnn_states,self.user_featEmb, self.selected_item_mask])
       active_user_state, active_rnn_states, active_userEmb, active_selected_item_mask = active_re
      

       if return_UN:
          return active_user_state, active_userEmb, active_rnn_states, active_selected_item_mask, self.appeared_users_num
       return active_user_state, active_userEmb, active_rnn_states, active_selected_item_mask


    def update(self, next_u_states, next_u_rnnSt, leave_mask, action):
       
       leave_mask = leave_mask.to(int)
       old_active = torch.nonzero(self.active_user_mask)
       
       self.active_user_mask[old_active] = leave_mask  # update 

   
       self.user_states[old_active.squeeze(dim=-1), :] = next_u_states
       self.rnn_states[old_active.squeeze(dim=-1), :] = next_u_rnnSt

       assert torch.sum(self.selected_item_mask) < 0.001

 

      



    

       

    def _init_users(self, pretrainRNN, T):
        
        if self.is_train:
            user_list_path = self.group_config['user_file']
        else:
            user_list_path = self.group_config['test_user_file']
        with open(user_list_path, 'rb') as f:
         user_feat = pickle.load(f)
        print('[UserGroup]%s load %d users from file %s'%(self.group_name, len(user_feat), user_list_path))

        #determine new user at each time $t$
        beta = int(self.group_config['beta'])
        inT = int(self.group_config['inT'])
        assert beta*inT < len(user_feat)
        user_time = [0 for _ in range(len(user_feat) -beta*(inT-1))]
        for t in range(1,inT):
           user_time.extend([t for _ in range(beta)])
        self.user_time_slot = torch.tensor(np.array(user_time), device=self.device)

        user_feats = torch.tensor(np.array(user_feat, dtype=np.int64), dtype=torch.int64, device=self.device) #[gU, 5]

        self.user_featEmb = pretrainRNN.get_user_featEmb(user_feats)
        self.user_states, self.rnn_states = utils.get_next_userState(self.user_featEmb, pretrainRNN) # rnn_states: [hx,cx]
        assert self.user_states.size()[0] == user_feats.size()[0]

        self.active_user_mask = (self.user_time_slot<1).to(dtype=torch.int64, device=self.device)
        self.appeared_users_num = torch.sum(self.active_user_mask)

        self.selected_item_mask = torch.zeros(self.user_featEmb.size()[0], pretrainRNN.item_count, dtype=torch.float64).to(self.device)
   






        
