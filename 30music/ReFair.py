import torch
from torch.optim import Adam
import utils
import json
import os
import gc


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




class ReFair:

    def __init__(self, policy_config, groups, phi_sa_normalize, device):
        super(ReFair, self).__init__()

        self.policy_config = policy_config
        self.needOptimize = True
        self.device = device
        print('[ReFair] policy config: ', dict(self.policy_config))

        # policy-related config 
        self.item_count = int(self.policy_config['item_count'])
        self.dim = int(self.policy_config['dim'])
        self.temp = float(self.policy_config['temp'])

        self.beta_3 = float(self.policy_config['beta3'])
        self.policy_epochs = json.loads(self.policy_config['policy_epochs'])
        self.pi_lr = float(self.policy_config['pi_lr'])
        self.c_lr = float(self.policy_config['c_lr'])
        self.clambda_max = float(self.policy_config['clambda_max'])
        self.w = float(self.policy_config['w'])
        self.lamda = float(self.policy_config['lambda'])
        self.v = float(self.policy_config['v'])
        self.S = int(self.policy_config['s'])
        self.eta = float(self.policy_config['eta'])
        self.Q_min_clip = float(self.policy_config['Q_min_clip'])
        self.Q_max_clip = float(self.policy_config['Q_max_clip'])



        self.small_number = 0.0000001
        self.phi_sa_normalize = phi_sa_normalize


         # init tensor 
        self.vartheta = torch.nn.Linear(self.dim, 1, device=self.device,bias=False, dtype=torch.float64)
       

        self.pi_opt = Adam(self.vartheta.parameters(), lr=self.pi_lr)
      




     
        
    def cal_group(self, mdp, u_states):

        with  torch.no_grad():
          phi_sa = torch.mul(u_states.unsqueeze(dim=1), mdp.qfunction.itemEmb.unsqueeze(0)) #[vU, 1, d] , [1, A, d] --> [vU, A, d] 
          if self.phi_sa_normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 

          # pred_retention
          ns_prob = torch.matmul(phi_sa, mdp.mu_t.t()) #[vU, A, d] *[d,2]--> [vU, A, 2]
          ns_zero =  torch.zeros_like(ns_prob)
          ns_prob = torch.maximum(ns_prob, torch.zeros_like(ns_prob)) + self.small_number
          ns = ns_prob / torch.sum(ns_prob, dim=-1, keepdim=True)  #[vU, A]
          ns_stay = ns[:, :, 0]


  


       

          # uncertainty
          un_2D = torch.sqrt(torch.sum(torch.mul(torch.matmul(phi_sa, mdp.inv_Lambda), phi_sa ), dim=-1 )) #[vU, A,d]*[d,d]--> [vU, A, d] cdot [vU, A, d] --> [vU, A, d] --> [vU, A]


           # Q_val
          Q_val_raw = mdp.qfunction.get_q_value(u_states, normalize=self.phi_sa_normalize) #[vU0+vU1, A]
          Q_val_mean = torch.mean(Q_val_raw, dim=-1, keepdim=True) #[vU0+vU1, 1]
          Q_val = Q_val_raw- Q_val_mean
       

          del ns_zero
          gc.collect()

        return phi_sa, Q_val, ns_stay, un_2D





    def pre_cal_Q_stay(self, mdp_l, groups, time):
         
        phi_sa_l, Q_val_l , stay_p_l, ct_l = [], [], [], []
        for gi in range(len(groups)):
            g = groups[gi]
            g_u_states,  _, _, _ = g.get_validUsers(time) #[vU, d], make active_user_mask updated

            g_phi_sa, g_Q_val , g_stay_p , g_ct = self.cal_group(mdp_l[gi], g_u_states)

            phi_sa_l.append(g_phi_sa.detach())
            Q_val_l.append(g_Q_val.detach())
            stay_p_l.append(g_stay_p.detach())
            ct_l.append(g_ct.detach())
       

    
        return phi_sa_l, Q_val_l, stay_p_l, ct_l
    

    def get_pi_prob(self, phi_sa_l):

         
        pi_prob_l = []
        for gi in range(len(phi_sa_l)):
           psa = phi_sa_l[gi]
           pi_prob_logits = self.vartheta(psa).squeeze(dim=-1)
           tmp_pi_prob = torch.nn.functional.softmax(pi_prob_logits/self.temp, dim=-1) #[vU, A]
           pi_prob_l.append(tmp_pi_prob)
        
       
        return pi_prob_l
    
    def sampling(self, pi_prob_l):
        sampled_action_l = []
        for gi in range(len(pi_prob_l)):
           g_sample_S = []
           g_dist = torch.distributions.categorical.Categorical(probs=pi_prob_l[gi])

           for si in range(self.S):
              sampled_a = g_dist.sample().unsqueeze(dim=-1) #[vU0,1]
              g_sample_S.append(sampled_a)
           sampled_action_l.append(torch.cat(g_sample_S, dim=-1))
        
        return sampled_action_l

    
    def update_lambda(self, stay_p_l, g_pi_prob, ct_l, g_re_rates):
        # calculate violation
        c_vio = self.cal_constraint_violation(stay_p_l, g_pi_prob, g_re_rates)

        g0_avg_ct = torch.mean(torch.sum(torch.mul(ct_l[0],g_pi_prob[0]),dim=-1))
        g1_avg_ct = torch.mean(torch.sum(torch.mul(ct_l[1],g_pi_prob[1]),dim=-1))
        ct= g0_avg_ct + self.w* (g_re_rates[1]/ g_re_rates[0])* g1_avg_ct
        ct= torch.square(self.beta_3*ct)

        c_vio_abs = c_vio*c_vio


        # print('[update_lamdba: ]', 'g0_avg_ct: ', g0_avg_ct.item(), 'g1_avg_ct', g1_avg_ct.item(), 'ct', ct.item(), 'c_vio', c_vio, 'c_vio_abs:', c_vio_abs, 'beta_3: ', self.beta_3)

        l_updates = (ct -c_vio_abs).detach()
        self.lamda = min([max([self.lamda - self.c_lr*l_updates, 0.0]), self.clambda_max])
  


    def cal_constraint_violation(self, stay_p_l,g_pi_prob, g_re_rates, is_print=False):
       
        g0_avg_stay = torch.mean(torch.sum(torch.mul(stay_p_l[0], g_pi_prob[0]), dim=-1))
        g1_avg_stay = torch.mean(torch.sum(torch.mul(stay_p_l[1], g_pi_prob[1]), dim=-1))


        c_vio =  (g0_avg_stay - (g_re_rates[1]/g_re_rates[0])*self.w*g1_avg_stay).detach()

        if g_re_rates[0] > g_re_rates[1]:
            c_vio = max(0, c_vio.item())
        elif g_re_rates[0] < g_re_rates[1]: 
            c_vio = min(0, c_vio.item())

        if is_print:
           print('g0_avg_stay: ', g0_avg_stay.item(), 'g1_avg_stay: ', g1_avg_stay.item(), 'c_vio:',c_vio)

        return c_vio


    

    def optimize(self, mdp_l, group_l, time, g_re_rates=[1.0,1.0]):
        # optimize with global pretrained_rnn
        phi_sa_l, Q_val_l, stay_p_l, ct_l = self.pre_cal_Q_stay(mdp_l, group_l, time)

        row_index = [torch.arange(Q_val_l[0].size()[0]).unsqueeze(dim=-1),torch.arange(Q_val_l[1].size()[0]).unsqueeze(dim=-1) ]
        
        # version for two groups
        for epochs in range(self.policy_epochs[0]):

          old_pi_prob_l = self.get_pi_prob(phi_sa_l)  #[[vU0, A ], [vU1, A]]
          old_pi_prob_l =[pe.detach() for pe in old_pi_prob_l]
          sampled_action_l = self.sampling(old_pi_prob_l) #[[vU0, S ], [vU1, S]]
        
          # updata lambda
          self.update_lambda(stay_p_l, old_pi_prob_l, ct_l, g_re_rates)

          for in_epoch in range(self.policy_epochs[1]):
            # calculate pi_prob
            g0_pi, g1_pi = self.get_pi_prob(phi_sa_l)
    

            c_vio = self.cal_constraint_violation(stay_p_l, [g0_pi.detach(), g1_pi.detach()], g_re_rates, is_print=False)


            pi_loss_2D_l, ratio_l, debug_lamba = [],[], []
          
            for ind in range(self.S):
                b_g0_sa, b_g1_sa = sampled_action_l[0][:,ind].unsqueeze(dim=-1), sampled_action_l[1][:,ind].unsqueeze(dim=-1)
                b_g0_sa_oldp, b_g1_sa_oldp = old_pi_prob_l[0][row_index[0], b_g0_sa], old_pi_prob_l[1][row_index[1], b_g1_sa]
               
            
                b_g0_Q = Q_val_l[0][row_index[0], b_g0_sa]
                b_g1_Q = Q_val_l[1][row_index[1], b_g1_sa]

                # calculate violation
                b_g0_stay = stay_p_l[0][row_index[0], b_g0_sa]
                b_g1_stay = stay_p_l[1][row_index[1], b_g1_sa]

                g0_c_derive = b_g0_stay
                g1_c_derive = b_g1_stay

            


            
                g0_updates = b_g0_Q - 2* self.lamda*c_vio*g0_c_derive #[vU0, 1]
                g1_updates = b_g1_Q + 2* self.lamda*(g_re_rates[1]/g_re_rates[0])*self.w*c_vio*g1_c_derive #[vU1, 1]
                b_updates = torch.cat([g0_updates, g1_updates], dim=0) #[vU0+vU1, 1]
                b_updates = torch.clip(b_updates,min=self.Q_min_clip, max=self.Q_max_clip)

                debug_lamba.append([[b_g0_Q, 2* self.lamda*c_vio*g0_c_derive, g0_updates, torch.clip(g0_updates,min=self.Q_min_clip, max=self.Q_max_clip)], [b_g1_Q,  2* self.lamda*(g_re_rates[1]/g_re_rates[0])*self.w*c_vio*g1_c_derive, g1_updates, torch.clip(g1_updates,min=self.Q_min_clip, max=self.Q_max_clip)]])


                b_g0_ratio = torch.div(g0_pi[row_index[0], b_g0_sa], b_g0_sa_oldp)
                b_g1_ratio = torch.div(g1_pi[row_index[1], b_g1_sa], b_g1_sa_oldp)
                b_ratio = torch.cat([b_g0_ratio, b_g1_ratio], dim=0) #[vU0+vU1, 1]
                b_pi_loss = torch.mul(b_updates, b_ratio)
                
                pi_loss_2D_l.append(b_pi_loss)
                ratio_l.append(b_ratio)
                

            ratio = torch.cat(ratio_l, dim=-1)
            # print('ratio: ', torch.mean(ratio).item(), torch.max(ratio).item(), torch.min(ratio).item())
           
            # # debug for lambda updates
            # for d_gi in range(2):
            #     debug_Q = torch.cat([e[d_gi][0] for e in debug_lamba], dim=-1)
            #     debug_vio_addi = torch.cat([e[d_gi][1] for e in debug_lamba], dim=-1)
            #     debug_u = torch.cat([e[d_gi][2] for e in debug_lamba], dim=-1)
            #     debug_u_clip = torch.cat([e[d_gi][3] for e in debug_lamba], dim=-1)
                
            #     print('[debug_lambda]:', debug_Q.size(), debug_vio_addi.size(), debug_u.size())
            #     print('[debug_lambda] gi:%d, debug_Q: '%d_gi, [torch.mean(debug_Q).item(), torch.max(debug_Q).item(), torch.min(debug_Q).item()])
            #     print('[debug_lambda] gi:%d, debug_vio_addi: '%d_gi, [torch.mean(debug_vio_addi).item(), torch.max(debug_vio_addi).item(), torch.min(debug_vio_addi).item()])
            #     print('[debug_lambda] gi:%d, debug_u: '%d_gi, [torch.mean(debug_u).item(), torch.max(debug_u).item(), torch.min(debug_u).item()])
            #     print('[debug_lambda] gi:%d, debug_u_clip: '%d_gi, [torch.mean(debug_u_clip).item(), torch.max(debug_u_clip).item(), torch.min(debug_u_clip).item()])
            
                
             
            
            
            


            # loss 
            loss_2D = torch.cat(pi_loss_2D_l, dim=-1)
            #print('loss_2D: ', torch.mean(loss_2D).item(), torch.max(loss_2D).item(), torch.min(loss_2D).item())

            pi_loss = torch.mean(torch.mean(loss_2D, dim=-1))

            old_pi_all = torch.cat(old_pi_prob_l, dim=0)
            g_pi_all = torch.cat([ g0_pi, g1_pi], dim=0)
            kl_loss =torch.mean(torch.sum(torch.mul(g_pi_all, torch.log(torch.div(g_pi_all, old_pi_all) + self.small_number)), dim=-1))

            loss = (kl_loss - 1.0/self.v *pi_loss) * (kl_loss.detach() <=self.eta).type(torch.float64) 

            #print('loss: ', loss.item(), 'kl_loss: ', kl_loss.item(), 'pi_loss', pi_loss.item(), 'v', self.v)
            self.pi_opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.vartheta.parameters(), 40)
            self.pi_opt.step()
            
            # Earyly stopping
            g_pi_new= self.get_pi_prob(phi_sa_l)
            g_pi_all_new = torch.cat(g_pi_new, dim=0)
            kl_loss =torch.mean(torch.sum(torch.mul(g_pi_all_new, torch.log(torch.div(g_pi_all_new, old_pi_all) + self.small_number)), dim=-1))
            if kl_loss > self.eta:
                print('[policy] Break at eopch {}-{} because of KL value {:.4f} larger than {:.4f}'.format(in_epoch,epochs, kl_loss, self.eta))
                break
        del phi_sa_l, stay_p_l, ct_l, row_index, old_pi_prob_l, sampled_action_l
        gc.collect()
        torch.cuda.empty_cache()






   
             
            





            











    def excute(self, u_states, itemEmb, argmax=True):

        phi_sa = torch.mul(u_states.unsqueeze(dim=1), itemEmb.unsqueeze(0)) #[vU, 1, d] , [1, A, d] --> [vU, A, d] 
        if self.phi_sa_normalize:
            phi_sa = torch.nn.functional.normalize(phi_sa, dim=-1) 
        
        pi_prob_logits = self.vartheta(phi_sa).squeeze(dim=-1)
        pi_prob = torch.nn.functional.softmax(pi_prob_logits/self.temp, dim=-1) #[vU, A]
        

        
        if argmax:
            action = torch.argmax(pi_prob, dim=-1, keepdim=False)
        else:
            #sample according to pi_prob
            prob_cumsum = pi_prob.cumsum(dim=-1)
            rand_p = torch.rand(pi_prob.size()[0],1, device=self.device)
            mask_2D = torch.sum((prob_cumsum <= rand_p), dim=-1)
            action = torch.min(mask_2D,self.item_count * torch.ones_like(mask_2D))

        return action











            
        
        


           









