
from tokenize import group
import torch
import pickle
import numpy as np
from torch.optim import Adam
import utils
import gc
import json 




class Qfunction:

    def __init__(self, mdp_config, pretrainRNN, device):

        self.mdp_config = mdp_config
        self.device = device
        self.dim = int(self.mdp_config['dim'])
        self.alpha = float(self.mdp_config['q_alpha'])
        self.item_count  = pretrainRNN.item_count
        self.itemEmb=pretrainRNN.item_tower(torch.arange(start=0, end=self.item_count, device=self.device))
        
        


        self.w = torch.nn.Linear(self.dim, 1, device=self.device,bias=False, dtype=torch.float64)  # [d,1] for calculate Q


        self.optimizer = Adam(self.w.parameters(), lr=self.alpha)




    


    def get_q_value(self, user_st, action=None, normalize=False):
        '''
          user_st : [vU, d]
          action: 
          -- None: return q values of all items, [vU, A]
          -- [vU, 1]: return [vU]
        '''
        
        if action is None:
            phi_sa = torch.mul(user_st.unsqueeze(dim=1), self.itemEmb.unsqueeze(dim=0))  #[vU,1, d] \times [1,A, d] --> [vU, A, d]
        else:
            actionEmb = self.itemEmb[action,:] #[vU, d]
            phi_sa = torch.mul(user_st, actionEmb) #[vU, d]

        if normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 


        Q_val = self.w(phi_sa).squeeze(dim=-1)
       
        return Q_val
    
    def update(self, delta):
        self.optimizer.zero_grad() 
        (delta ** 2).mean().backward()
        self.optimizer.step()

    


class MDP:
    def __init__(self, mdp_config,T, pretrainRNN, group_ind, device='cuda'):

        self.mdp_config = mdp_config
        self.group_ind = group_ind

        self.dim = int(self.mdp_config['dim'])
        self.T = T
        self.H = int(self.mdp_config['h'])
        self.b_beta = float(self.mdp_config['b_beta'])
        self.epsilon = float(self.mdp_config['epsilion'])
        self.K = int(self.mdp_config['k'])
        self.time = 0
        self.device = device
        self.reward_thre = float(self.mdp_config['reward_thre'])
        self.leave_state = int(self.mdp_config['leave_state'])
        self.init_lambda = int(self.mdp_config['lambda'])
        self.gamma = float(self.mdp_config['gamma'])
        self.warmT = float(self.mdp_config['warmT'])
        self.r_offset = json.loads(self.mdp_config['rewards_offsets'])[group_ind]

        print('[init MDP] group %d, r_offset: %d'%(group_ind, self.r_offset))

        self.item_count = pretrainRNN.item_count

        self.qfunction = Qfunction(mdp_config, pretrainRNN, device=self.device)
        self.debug_info = {}


        


        # MDP
        self.theta_t = torch.rand((self.dim,1), device=self.device, requires_grad=False, dtype=torch.float64)
        self.r_phi = torch.zeros(self.dim,1, device=self.device, requires_grad=False, dtype=torch.float64)

        self.mu_t = torch.rand((2, self.dim), device=self.device, requires_grad=False, dtype=torch.float64)
        self.sigma_phi = torch.zeros(2, self.dim, device=self.device, requires_grad=False, dtype=torch.float64)

        self.Lambda = self.init_lambda*torch.eye(self.dim,  device=self.device,requires_grad=False, dtype=torch.float64)
        self.inv_Lambda = torch.linalg.inv(self.Lambda)
        self.Z = torch.linalg.det(self.Lambda)

        

 

 
    def sample_select_action(self, valid_st, selected_item_mask, sampled_stay_mask, phi_sa_normaize,  policy_name='greedy', policy_obj=None):

       
        if policy_name == 'greedy':
           # epsilion-greedy
           Q_val = self.qfunction.get_q_value(valid_st, normalize=phi_sa_normaize) #[vU, A]
           assert torch.abs(torch.sum(Q_val[torch.nonzero((1-sampled_stay_mask.squeeze(-1))).squeeze(-1), :]))  < 0.0001 # Q(o,a)=0, for all a

           Q_val = Q_val - selected_item_mask

           max_q_value, argmax_a = torch.max(Q_val, dim=-1) #[vU, 1]
       

           
        else:
            # randomly sample according to previous policy 
            argmax_a = policy_obj.excute(valid_st, self.qfunction.itemEmb, argmax=False)
          

        random_p = torch.rand(argmax_a.size(), device=self.device)

        random_a = torch.randint(low=0, high=self.item_count, size=argmax_a.size(), device=self.device)

        select_a = torch.where(random_p<=self.epsilon, random_a, argmax_a)


        
        return select_a


    def sample_execute(self, u_states, action, pretrainRNN):

        # get predictions
        # look up actions itemEmb
        actEmb = self.qfunction.itemEmb[action, :] #[vU, d]
        ui_emb = torch.mul(u_states, actEmb) #[vU, d]

        if pretrainRNN.phi_sa_normalize:
            ui_emb = torch.nn.functional.normalize(ui_emb, dim=-1)

        r_hat = torch.matmul(ui_emb, self.theta_t) #[vU, 1]
        zeros = torch.zeros_like(r_hat)
        ones = torch.ones_like(r_hat)
        r_hat_cap = torch.min(torch.max(zeros, r_hat), ones)

        ns = torch.matmul(ui_emb, self.mu_t.t()) #[vU, S]
        b_t = self.b_beta * torch.sqrt(torch.sum( torch.mul(torch.matmul(ui_emb, self.inv_Lambda), ui_emb),dim=-1, keepdim=True)) #[vU,1]
        r_tilde = r_hat_cap + b_t  # check range

     
        
        # normalize ns to [0,1]
        #ns = torch.maximum(zeros, ns)
        ns = torch.maximum(torch.zeros_like(ns), ns) + 0.0000001
        ns = ns / torch.sum(ns, dim=-1, keepdim=True)

        assert torch.sum(ns.isnan()) < 0.1

    
        


        rand_p = torch.rand(ns.size()[0],1, device=self.device)
        cumporb = torch.cumsum(ns, dim=-1)
        mask_2D = torch.sum((cumporb <= rand_p), dim=-1, keepdim=True)
        mask_2d_ones = torch.ones_like(mask_2D)
        next_state = torch.min(mask_2D, self.leave_state*mask_2d_ones )
        


        return r_tilde, next_state


    def update_theta_mu(self):

        self.inv_Lambda = torch.linalg.inv(self.Lambda)
        self.theta_t = torch.matmul(self.inv_Lambda, self.r_phi)
        self.mu_t = torch.matmul(self.sigma_phi, self.inv_Lambda)





    def update_mdp_stat(self, u_states, action, reward, next_state, normalize):

        actEmb = self.qfunction.itemEmb[action, :] #[vU, d]
        phi_sa = torch.mul(u_states, actEmb) # [vU,d]

        if normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 





        self.Lambda +=  torch.sum(torch.matmul(phi_sa.unsqueeze(-1), phi_sa.unsqueeze(1)), dim=0)   #[vU,d, 1] * [vU,1, d] -->[vU, d,d] --> [d,d]
        self.r_phi += torch.sum(torch.mul(phi_sa,reward), dim=0).unsqueeze(-1)
        

        # next_state
          
        batch_size = next_state.size()[0]
        sigma = torch.zeros(batch_size, 2, dtype=torch.float64,device=self.device)
        sigma[torch.arange(batch_size).unsqueeze(-1),next_state] =1 #[vU,2]

        self.sigma_phi +=  torch.sum(torch.matmul(sigma.unsqueeze(-1), phi_sa.unsqueeze(1) ), dim=0)#[vU, 2, 1] * [vU, 1, d] --> [vU, 2,d]
        
      


        
 

    

    def get_state_value(self, states, phi_sa_normalize):

        # V(s) = max_a Q(s,a)
        Q_value = self.qfunction.get_q_value(states, normalize=phi_sa_normalize) #[vU, A]

        state_value, _ = torch.max(Q_value, dim=-1) #[vU,1]

        return state_value

    def calQ(self, t_group, pretrainRNN, time, policy_name='greedy', policy_obj=None):

        # update theta_t, mu_t
        self.time=time
        self.update_theta_mu()
        self.Z = torch.linalg.det(self.Lambda)



        # updates Q-function based on samples on M_t        
        u_states,  u_featEmb, u_rnnSt,  u_sItem= t_group.get_validUsers(self.time) #[vU, d]

            
        # if the item already be selected, set it to 1e30
        sampled_item_mask = u_sItem.clone().detach()
        sampled_stay_mask = torch.ones(u_states.size()[0], 1, dtype=torch.int64).to(self.device) # 1 for stay; 0 for leave
            
        # sample based on current MDP
        step = 0 
        iteration_step = self.H
        num_active_users = torch.sum(sampled_stay_mask)
        while (step < iteration_step) and (num_active_users >0):
               # select action with episilon-greedu
               s_action = self.sample_select_action(u_states, sampled_item_mask, sampled_stay_mask, phi_sa_normaize=pretrainRNN.phi_sa_normalize, policy_name=policy_name, policy_obj=policy_obj) #[vU,1]

               # reward: \hat{r}_t + b_t, transition: \mu_t
               reward, ns_ind = self.sample_execute(u_states, s_action, pretrainRNN)

               # record interaction
               sampled_item_mask[torch.arange(s_action.size()[0]).unsqueeze(-1), s_action.unsqueeze(-1)] = 1e30

               # left users
               leave_mask = 1-(ns_ind==self.leave_state).to(torch.float64)
               next_sampled_stay_mask = torch.mul(sampled_stay_mask, leave_mask) #[vU, 1]
               
             
               # generate emb of next states
               sample_interactions = torch.concat([s_action.unsqueeze(-1),(reward >= self.reward_thre).to(int)], dim=-1) #[vU,2]
               next_u_states, next_u_rnnSt = utils.get_next_userState(u_featEmb, pretrainRNN, interactions=sample_interactions, user_histSeq=u_rnnSt)
               ## set states of leave users as zeros
               u_states_zeros =  torch.zeros_like(next_u_states)
               u_rnnstaes_zeros =  torch.zeros_like(next_u_rnnSt)
               next_u_states = torch.where(next_sampled_stay_mask>0, next_u_states, u_states_zeros)
               next_u_rnnSt = torch.where(next_sampled_stay_mask>0, next_u_rnnSt, u_rnnstaes_zeros)




               # do LSTD 
               q_value = self.qfunction.get_q_value(u_states.detach(), s_action, normalize=pretrainRNN.phi_sa_normalize) #[vU, 1]
               assert torch.abs(torch.sum(q_value[(sampled_stay_mask.squeeze(-1)==0).nonzero()])) < 0.0001 # Q(o,a)=0

               state_value = self.get_state_value(next_u_states, phi_sa_normalize=pretrainRNN.phi_sa_normalize).detach()
               assert torch.abs(torch.sum(state_value[(next_sampled_stay_mask.squeeze(-1)==0).nonzero()])) < 0.0001 # V(o) =0

               delta = reward.squeeze(-1).detach() + self.gamma*state_value - q_value
               self.qfunction.update(delta)




               # update
               u_states = next_u_states
               u_rnnSt = next_u_rnnSt
               sampled_stay_mask = next_sampled_stay_mask
               num_active_users = torch.sum(sampled_stay_mask)
               step +=1

              
        print('[calQ] %s iteration %d steps'%(t_group.group_name, step))

        del sampled_item_mask
        del sampled_stay_mask
        gc.collect()
        torch.cuda.empty_cache()
        


               


    def need_calQ(self, time_step):
        
        if time_step == 0:
            return True 
        
        det_L = torch.linalg.det(self.Lambda)
        print('need_calQ: ', det_L, 2* self.Z)
        if det_L > 2*self.Z:
            return True
        return False 
    
    def greedy_policy(self, u_states, u_selected_item, phi_sa_normalize):
        # calculate Q value:

        Q_value = self.qfunction.get_q_value(u_states,normalize=phi_sa_normalize) #[vU, A]
        Q_value = Q_value - u_selected_item

        action = torch.argmax(Q_value, dim=-1, keepdim=False)

        return action 
    
    def excute_policy(self, u_states, u_selected_item,  phi_sa_normalize, policy_name, policy_obj, u_active_mask=None):

       if policy_name == 'greedy':
          return self.greedy_policy(u_states, u_selected_item,  phi_sa_normalize)
       else:
          s_action = policy_obj.excute(u_states, itemEmb=self.qfunction.itemEmb, argmax=True)
          return s_action
      
       
    def real_execute(self, u_states, action, pretrainRNN):
        # generate reward/ next states from real simulator 
        
        act_emb = self.qfunction.itemEmb[action, :]
        r_raw, ns_prob = pretrainRNN.getPred(u_states,act_emb)
        r= r_raw - self.r_offset

        if pretrainRNN.use_CE_Nsloss:
            ns_prob = torch.nn.functional.softmax(ns_prob, dim=-1)
        
        assert torch.sum(ns_prob.isnan()) < 0.001
      

        assert torch.abs(torch.sum(torch.sum(ns_prob, dim=-1)-1 )) < 0.0001


        rand_p = torch.rand(ns_prob.size()[0],1, device=self.device)
        cumporb = torch.cumsum(ns_prob, dim=-1)
        mask_2D = torch.sum((cumporb <= rand_p), dim=-1, keepdim=True)
        mask_2D_ones = torch.ones_like(mask_2D)
        next_state = torch.min(mask_2D, self.leave_state* mask_2D_ones)



   

    
        return r, next_state

    

            

    def act(self, t_group, t_pretrainRNN, time, policy_name='greedy', policy_obj=None):
        '''
          choose action for each groups and return rewards
        '''
        
        tg_reward = 0
        tg_retention = [0, 0, 0] #[# user at time t, #user at time t-1]

       

        
           
        u_states,  u_featEmb, u_rnnSt, u_sItem, appear_uNum = t_group.get_validUsers(time, return_UN=True) #[vU, d]
        u_active_mask = t_group.active_user_mask
        #print('[act] user_num %d in group %s'% (u_states.size()[0], g.group_name))

        tg_retention[0] = u_states.size()[0]
        tg_retention[2] = appear_uNum.item()

        if (policy_name!='greedy') and (time==0):
                action = self.excute_policy(u_states, u_selected_item=u_sItem, phi_sa_normalize=t_pretrainRNN.phi_sa_normalize,  policy_name='greedy', policy_obj=None)
        else:
                action = self.excute_policy(u_states, u_selected_item=u_sItem, phi_sa_normalize=t_pretrainRNN.phi_sa_normalize,   policy_name=policy_name, policy_obj=policy_obj, u_active_mask=u_active_mask)

        reward, next_state = self.real_execute(u_states, action, t_pretrainRNN)

        # log debug 
        debug_ar_l = [u_states, action, torch.multiply(reward, 1-(next_state==self.leave_state).to(torch.float64)) ]


            

        tg_reward = np.power(self.gamma, self.time) * torch.sum(reward).item()

        # updates user states, according to the real excution.
        if time < self.warmT:
               print('wart stage: %d'%time)
               leave_mask = torch.ones_like(next_state)
        else:
               leave_mask = 1-(next_state==self.leave_state).to(torch.float64) # 1 for stay, 0 for leave
        tg_retention[1] = torch.sum(leave_mask).item()
            

        sample_interactions = torch.concat([action.unsqueeze(-1),(reward >= self.reward_thre).to(int)], dim=-1) #[vU,2] 
        next_u_states, next_u_rnnSt = utils.get_next_userState(u_featEmb, t_pretrainRNN, interactions=sample_interactions, user_histSeq=u_rnnSt)
            # set next_u_states & next_u_rnnSt of leave users as zero vector
            
        next_u_states_zeros =  torch.zeros_like(next_u_states)
        next_u_rnnSt_zeros = torch.zeros_like(next_u_rnnSt)
        next_u_states = torch.where(leave_mask>0, next_u_states, next_u_states_zeros)
        next_u_rnnSt = torch.where(leave_mask>0, next_u_rnnSt, next_u_rnnSt_zeros)

        t_group.update(next_u_states, next_u_rnnSt, leave_mask, action)

        # update MDP estimates
        self.update_mdp_stat(u_states, action, reward, next_state, normalize=t_pretrainRNN.phi_sa_normalize)

           
      
        return tg_reward, tg_retention, debug_ar_l



            



            


        
    

