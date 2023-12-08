
import torch
import json
import numpy as np
torch.set_printoptions(profile="full")



class pretrainRNNModel(torch.nn.Module):

    def __init__(self, model_config, item_feats, device='cuda',train_batch_size=512):
        super(pretrainRNNModel, self).__init__()

        self.model_config = model_config
        
        self.device = torch.device(device)
        self.dim = int(model_config['EMB_DIM'])
        self.nS = int(model_config['mu_dim'])
        uf_count = json.loads(model_config['uf_count'])
        self.uFeat = len(uf_count) # including uid
        self.train_batchSize = train_batch_size
        h0, c0 = torch.zeros(train_batch_size, self.dim, dtype=torch.float64).to(self.device),\
                torch.zeros(train_batch_size, self.dim, dtype=torch.float64).to(self.device)
        self.rnn_initial_states = (h0, c0)
        self.item_count = int(model_config['item_count'])


        # modules 
        self.userID_embs = torch.nn.Embedding(int(model_config['user_count']), self.dim, device=self.device, dtype=torch.float64) #[U, d]
        self.user_feat_embs = torch.nn.ModuleList([])
        for i in range(1, len(uf_count)):
            self.user_feat_embs.append(torch.nn.Embedding(uf_count[i], self.dim, device=self.device, dtype=torch.float64))

        self.itemID_embs = torch.nn.Embedding(int(model_config['item_count']), self.dim, device=self.device, dtype=torch.float64) #[I,d]
        self.item_cat_embs = torch.nn.Embedding(int(model_config['item_cat']), self.dim, device=self.device, dtype=torch.float64) #[C, d]
        self.item_feats = torch.tensor(item_feats, device=self.device, requires_grad=False, dtype=torch.float64) # [item cat mapping]


        ## RNN
        self.reward_embs = torch.nn.Embedding(2, self.dim, device=self.device) #[0/1, dim]
        self.next_state_embs = torch.nn.Embedding(self.nS, self.dim, device=self.device, dtype=torch.float64) #[3,dim]
        



        self.rnn =  torch.nn.LSTMCell(2*self.dim, self.dim, device=self.device, dtype=torch.float64)
        self.user_linear_proj = torch.nn.Linear((len(uf_count)+1)*self.dim, self.dim, bias=False, device=self.device, dtype=torch.float64) #[uF+seq --> dim]
        self.item_linear_proj = torch.nn.Linear(2*self.dim, self.dim, bias=False, device=self.device, dtype=torch.float64) #[itemID + cat --> dim]
        self.final_proj = torch.nn.Linear(self.dim, 1, bias=False, device=self.device, dtype=torch.float64)


        # loss criterion
        ns_wei=torch.tensor(np.array(json.loads(model_config['ns_wei']), dtype=np.float64), device=self.device)
        self.ns_wei=ns_wei.to(device=self.device, dtype=torch.float64)
        self.ns_loss_weight = float(self.model_config['ns_loss_wei'])
        self.loss_reg_neg = float(self.model_config['loss_reg_neg'])
        self.loss_reg_1sum = float(self.model_config['loss_reg_1sum'])

        self.use_CE_Nsloss = int(self.model_config['use_ce_nsloss'])
        self.phi_sa_normalize = int(self.model_config['phi_sa_normalize'])

        if self.use_CE_Nsloss:
        # loss criterion
           print('[model] use_CE_Nsloss!')
           self.r_criterion = torch.nn.MSELoss('mean')
           self.ns_criterion = torch.nn.CrossEntropyLoss(weight=self.ns_wei, reduction='mean')



        
        self.global_step = torch.tensor(0, device=self.device, dtype=torch.int64)

    def weighted_mse_loss(self, input, target, weight=None):

        if weight is None:
            return ((input-target)**2).mean()
        else:
            return (weight* (input-target)**2).mean()


  
    def getPred(self, user_state, item_embs, valid_bs=None):
        ui_emb_raw = torch.mul(user_state, item_embs) #[B,d]

        if self.phi_sa_normalize:
           ui_emb = torch.nn.functional.normalize(ui_emb_raw, dim=-1)
           assert torch.abs(torch.sum(torch.norm(ui_emb, dim=-1)-1)) < 0.0001
        else:
            ui_emb = ui_emb_raw
        
        # rating prediction 
        rating_logits = self.final_proj(ui_emb)

        # next state preds 
        ns_logits = torch.matmul(ui_emb, torch.t(self.next_state_embs.weight)) #[B,nS]

        if valid_bs is None:
            return rating_logits, ns_logits
        
        return  rating_logits[0:valid_bs,:], ns_logits[0:valid_bs,:]


    def forward(self, batch_samples, valid_bs, summarywriter=None):
        user_feats, item_IDs, label, hist_seq, seq_len, ns = batch_samples 
        
        user_state = self.user_tower(user_feats, hist_seq, seq_len) #[B,d]
        item_embs = self.item_tower(item_IDs) #[B,d]
      
        rating_logits, ns_logits = self.getPred(user_state, item_embs, valid_bs=valid_bs)


        if summarywriter is not None:
            # check whether ns_logits is postive; 
            neg_ns_pred = torch.min(torch.zeros_like(ns_logits, device=self.device), ns_logits)
            neg_num = torch.count_nonzero(neg_ns_pred)
            summarywriter.add_scalar('ns_pred/neg_num', neg_num, self.global_step)
            summarywriter.add_scalar('ns_pred/avg_neg_val', torch.sum(neg_ns_pred)/neg_num, self.global_step)

            # check \sum_s P(s'|s,a) =1
            prob_sum = torch.sum(ns_logits, dim=-1,  keepdim=True)
            summarywriter.add_scalar('ns_pred/x_sum', torch.mean(prob_sum),  self.global_step)
            summarywriter.add_histogram('ns_pred/x_sum_dist', prob_sum,  self.global_step)

            # rating logtis
            summarywriter.add_scalar('r_pred/rating', torch.mean(rating_logits),  self.global_step)
            summarywriter.add_histogram('r_pred/rating', rating_logits,  self.global_step)

            for tmpi in range(2):
              summarywriter.add_scalar('ns_pred/logits_%d'%tmpi, torch.mean(ns_logits[:,tmpi]),  self.global_step)
              summarywriter.add_histogram('ns_pred/logits_%d'%tmpi, ns_logits[:,tmpi],  self.global_step)

            

             


        
        return  rating_logits, ns_logits 
    
    def cal_loss(self, pred, r_label, ns_label, summarywriter=None):
        rating_pred, ns_pred = pred
        batch_size = rating_pred.size()[0]
        
        if self.use_CE_Nsloss:
            rating_pred_loss = self.r_criterion(rating_pred, r_label.unsqueeze(-1).to(torch.float64))

        else:
            rating_pred_loss = self.weighted_mse_loss(rating_pred, r_label.unsqueeze(-1).to(torch.float64))

        if self.use_CE_Nsloss:
            ns_logits = ns_pred
            ns_pred_loss = self.ns_criterion(ns_logits, ns_label)
            ns_pred = torch.nn.functional.softmax(ns_pred, dim=-1)

        else:
           # ns_label [B,1]
           weights = self.ns_wei[ns_label.to(int)].unsqueeze(-1)
        
           ns_label_2D = torch.zeros(batch_size, 2, device=self.device)
           ns_label_2D[torch.arange(batch_size).unsqueeze(-1), ns_label.unsqueeze(-1)] =1
           ns_pred_loss = self.weighted_mse_loss(ns_pred, ns_label_2D, weight=weights) # TODO check dim

      
        # add regularization to ensure P(s'|s,a)>0 & \sum_{s'} P(s'|s_a)=1.
        neg_ns_pred = torch.min(torch.zeros_like(ns_pred, device=self.device), ns_pred)
        neg_num = torch.count_nonzero(neg_ns_pred)
        reg_neg_loss = self.loss_reg_neg*torch.where(neg_num>0,torch.sum(neg_ns_pred)/neg_num, 0)
        
        prob_sum = torch.sum(ns_pred, dim=-1,  keepdim=True)
        ones = torch.ones(batch_size,1, device=self.device)
        reg_1sum_loss = self.loss_reg_1sum*self.weighted_mse_loss(prob_sum, ones)

        loss_info= {'rating_mse':rating_pred_loss, 'ns_mse': self.ns_loss_weight* ns_pred_loss, 'ns_reg_neg_loss':reg_neg_loss, 'ns_reg_1sum_loss':reg_1sum_loss}
        
        # printout some debug info
        pos_r_index = torch.nonzero(r_label).squeeze(-1)
        neg_r_index = torch.nonzero(1-r_label).squeeze(-1)

        pos_r_pred = torch.mean(rating_pred[pos_r_index])
        neg_r_pred =  torch.mean(rating_pred[neg_r_index])

        # ns
        ns_label_1_count = torch.sum(ns_label)
        leave_ind = torch.nonzero(ns_label).squeeze(-1)
        avg_leval_ns_pred =  torch.mean(ns_pred[leave_ind, :], dim=0)
        leave_ns_pred = torch.where(ns_label_1_count>0, avg_leval_ns_pred, torch.zeros_like(avg_leval_ns_pred))

        stay_ind = torch.nonzero(1-ns_label).squeeze(-1)
        stay_ns_pred = torch.mean(ns_pred[stay_ind, :], dim=0)

        debug_info = {'r_logit_mean': torch.mean(rating_pred.to(torch.float64)), 'r_label_mean': torch.mean(r_label.to(torch.float64)),
                      'pos_r_pred_mean': pos_r_pred, 'neg_r_pred_mean': neg_r_pred,
                     'leave_ns_pred_0':leave_ns_pred[0], 'leave_ns_pred_1':leave_ns_pred[1], 'stay_ns_pred_0': stay_ns_pred[0],'stay_ns_pred_1': stay_ns_pred[1],
                     'ns_neg_num':neg_num,
                     'ns_sum': torch.mean(prob_sum)
                     }
        
        loss_sum = rating_pred_loss + self.ns_loss_weight* ns_pred_loss + reg_neg_loss + reg_1sum_loss

        if summarywriter is not None:
            for lt, lv in loss_info.items():
                summarywriter.add_scalar('loss/%s'%lt, lv,  self.global_step)
            summarywriter.add_scalar('loss/all', loss_sum,  self.global_step)
        
        
        return loss_sum, loss_info, debug_info
    
    def cal_metric(self, pred, r_label, ns_label, r_metrics, ns_metrics):

        r_pred, ns_pred  = pred

        if self.use_CE_Nsloss:
            ns_pred = torch.nn.functional.softmax(ns_pred, dim=-1)
     

        for m, rm in r_metrics.items():
            r_metrics[m].update(r_pred.squeeze(dim=-1), r_label)

        for m, ns_m in ns_metrics.items():
            ns_metrics[m].update(ns_pred,ns_label)

        return r_metrics, ns_metrics






    
    def item_tower(self, itemIDs):


        b_item_id_emb = self.itemID_embs(itemIDs.to(torch.int64)) #[B,d]
        #print('item_feats', self.item_feats.size(), itemIDs.max(), itemIDs.min())

        b_item_feats = self.item_feats[itemIDs,:] #TODO: double check
        
        b_item_cat_emb = torch.matmul(b_item_feats, self.item_cat_embs.weight) #[B, C]*[C,d]--> [B,d]


        b_item_featEmb = torch.concat([b_item_id_emb, b_item_cat_emb], axis=-1) 
        
        b_item_out = self.item_linear_proj(b_item_featEmb) #[B,d]
        return  b_item_out


    def user_tower(self, user_feats, hist_seq, seq_len):
        
        
        user_featEmb = self.get_user_featEmb(user_feats)
        self.user_featEMB_out = user_featEmb


        # deal with sequence features 
        hx, cx = self.rnn_initial_states
        seq_outputs = [hx]
        max_seq_len = hist_seq.size()[1]
       


        # seq emb
        item_ind = hist_seq[:,:,0]
        s_itemEmb = self.item_tower(item_ind) #[B, max_len, d]
        item_r = hist_seq[:,:,1]
        s_rewardEmb = self.reward_embs(item_r)
        s_sEmb = torch.concat([s_itemEmb, s_rewardEmb], axis=-1) #[B, max_len, 2d]
     
        for i  in range(max_seq_len): # max_seq_len
            hx, cx = self.rnn(s_sEmb[:,i,:], (hx,cx))
            seq_outputs.append(hx)

        seq_out = torch.stack(seq_outputs, dim=1) #[B, max_len, d]

        seq_len = seq_len.unsqueeze(dim=-1) 
        select_t = seq_len.expand(-1, self.dim).unsqueeze(dim=1) #[B, 1, d]
        seq_out = torch.gather(seq_out, dim=1, index=select_t).squeeze(1) #[B,d]

        

        user_Emb_c = torch.concat([user_featEmb, seq_out], axis=-1)  # #[B, (uFeat+1)*d]
        final_user_emb = self.user_linear_proj(user_Emb_c) #[B,d]
        return final_user_emb
    

    def get_user_featEmb(self, user_feats):

        user_featEmb_l = []
        for i in range(self.uFeat):
            u_Fid = user_feats[:, i] # TODO check shape
            if i == 0:
                user_featEmb_l.append(self.userID_embs(u_Fid))
            else:
                user_featEmb_l.append(self.user_feat_embs[i-1](u_Fid))
        user_featEmb = torch.concat(user_featEmb_l, axis=-1)

        return user_featEmb
    
    def get_next_userState(self, user_featEmb, interactions=None, user_histSeq = None):
        '''
          user_featEmb: [B, d]
          interactions: [B,2], [item_id, feedback]
          user_preSeq: [B, 2d] [hx,cx]
        '''

        if user_histSeq is None:
            # no historical interactions
            hx, cx = self.rnn_initial_states
        else:
            hx, cx = torch.split(user_histSeq, self.dim, dim=1) 

        if interactions is None:
            # inital state
            n_hx, n_cx = self.rnn_initial_states
        else:

            # embedding items:
            item_ind = interactions[:,0]
            s_itemEmb = self.item_tower(item_ind) #[B, d]
            item_r = interactions[:,1]
            s_rewardEmb = self.reward_embs(item_r) #[B,d]
            s_sEmb = torch.concat([s_itemEmb, s_rewardEmb], axis=-1) #[B, 2d]

            n_hx, n_cx = self.rnn(s_sEmb, (hx,cx))
        

        # get user_new state
        user_featEmb = torch.concat([user_featEmb, n_hx], axis=-1)
        final_user_emb = self.user_linear_proj(user_featEmb)

        rnn_hcx = torch.cat([n_hx, n_cx], dim=-1)

        return final_user_emb, rnn_hcx
     



    












        
        





