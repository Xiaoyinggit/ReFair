import torch
import pandas as pd
from contextlib import ContextDecorator
import gc

def get_n_tensors():
     tensors= []
     for obj in gc.get_objects():
      try:
         if (torch.is_tensor(obj) or
         (hasattr(obj, 'data') and
         torch.is_tensor(obj.data))):
             tensors.append(obj)
      except:
          pass
      return len(tensors)

class check_memory_leak_context(ContextDecorator):
    def __enter__(self):
        self.start = get_n_tensors()
        return self    
    
    def __exit__(self, *exc):
         self.end = get_n_tensors()
         increase = self.end - self.start
         
         if increase > 0:
              print("num tensors increased with ", self.end - self.start)
         else:
              print("no added tensors")
         return False

def do_padding(batch_samples, device, batch_size):
    user_feats, itemIDs, label, hist_seq, seq_len, ns = batch_samples 
        
    # padding users
    user_feats = user_feats.to(device) #[B, uFeat]
    hist_seq = hist_seq.to(device) #[B, max_len, 2]
    seq_len = seq_len.to(device) # [B,1]
    itemIDs = itemIDs.to(device)
    label = label.to(device) #[B,1]
    ns = ns.to(device) #[B,1]



    valid_bs = user_feats.size()[0]
    #valid_mask = torch.arange(batch_size) <  valid_bs

    padding_num = batch_size-valid_bs
    if padding_num >0:
        user_feats = torch.concat([user_feats, torch.zeros(padding_num, user_feats.size()[1],device=device, dtype=torch.int64)], dim=0)
        hist_seq = torch.concat([hist_seq, torch.zeros(padding_num, hist_seq.size()[1], hist_seq.size()[2],device=device, dtype=torch.int64)], dim=0)
        seq_len = torch.concat([seq_len, torch.zeros(padding_num, device=device, dtype=torch.int64)], dim=0)
        
        itemIDs = torch.concat([itemIDs, torch.zeros(padding_num, device=device, dtype=torch.int64)], dim=0)
    
    return [user_feats, itemIDs, label, hist_seq, seq_len, ns], valid_bs



def get_next_userState(userFeatEmbs, pretrainRNN, interactions=None, user_histSeq=None):

    # divide into batches
    batch_size = pretrainRNN.train_batchSize
    num_U = userFeatEmbs.size()[0] #[gU, uF*d]

    batch_num = num_U//batch_size
    st, ed = 0, batch_size
    
    u_states_l , u_rnn_states_l = [], []
    for bi in range(batch_num):
        batch_uFeatEmbs = userFeatEmbs[st:ed, :]
        batch_interactions = None
        if interactions is not None:
            batch_interactions = interactions[st:ed,:]
        batch_user_histSeq = None
        if user_histSeq is not None:
            batch_user_histSeq = user_histSeq[st:ed, :]
        
        assert (ed-st) == batch_size

        b_user_states, b_rnn_states = pretrainRNN.get_next_userState(batch_uFeatEmbs, interactions=batch_interactions, user_histSeq=batch_user_histSeq)
        u_states_l.append(b_user_states)
        u_rnn_states_l.append(b_rnn_states)
        st, ed = st+batch_size, ed+batch_size

    # last batch
    batch_uFeatEmbs = userFeatEmbs[st:, :]
    batch_interactions = None
    if interactions is not None:
        batch_interactions = interactions[st:,:]
    batch_user_histSeq = None
    if user_histSeq is not None:
        batch_user_histSeq = user_histSeq[st:,:]
    # do padding 
    valid_bs = batch_uFeatEmbs.size()[0]
    batch_uFeatEmbs = torch.concat([batch_uFeatEmbs, torch.zeros(batch_size-valid_bs, batch_uFeatEmbs.size()[1], device=batch_uFeatEmbs.device, dtype=batch_uFeatEmbs.dtype)], dim=0)
    if batch_interactions is not None:
        batch_interactions = torch.concat([batch_interactions, torch.zeros(batch_size-valid_bs, batch_interactions.size()[1], device=batch_interactions.device, dtype=batch_interactions.dtype)], dim=0)
    if batch_user_histSeq is not None:
        batch_user_histSeq = torch.concat([batch_user_histSeq, torch.zeros(batch_size-valid_bs, batch_user_histSeq.size()[1], device=batch_user_histSeq.device, dtype=batch_user_histSeq.dtype)], dim=0)
    b_user_states, b_rnn_states = pretrainRNN.get_next_userState(batch_uFeatEmbs, interactions=batch_interactions, user_histSeq=batch_user_histSeq)
  
    u_states_l.append(b_user_states[0:valid_bs, :])
    u_rnn_states_l.append(b_rnn_states[0:valid_bs, :])

    u_states = torch.cat(u_states_l, dim=0)
    rnn_states = torch.cat(u_rnn_states_l, dim=0)
    
    assert u_states.size()[0] == userFeatEmbs.size()[0]
    return u_states, rnn_states


def save_re(reward_l, retention_l, prefix, group_num):

    reward_df = pd.DataFrame(reward_l, columns=['r_g%d'%i for i in range(group_num)])
    retention_df = pd.DataFrame(retention_l, columns=['r_g%d_re'%i for i in range(group_num)])
    for i in range(group_num):
       retention_df['uin_g%d'%i] = retention_df['r_g%d_re'%i].map(lambda x: x[0])
       retention_df['uout_g%d'%i] = retention_df['r_g%d_re'%i].map(lambda x: x[1])
       retention_df['uappear_g%d'%i] = retention_df['r_g%d_re'%i].map(lambda x: x[2])
       retention_df['retention_g%d'%i] = retention_df['uout_g%d'%i]/retention_df['uappear_g%d'%i]
    # save df 
    reward_df.to_csv(prefix+'_reward.csv')
    retention_df.to_csv(prefix+'_retention.csv')

def debug_top_k(debug_matrix_dict, top_k_tuple):

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    K=5
    top_k_ind = top_k_tuple.indices
    for mn, m in debug_matrix_dict.items():
        print('--: ', mn)
        d_re = []
        for i in range(K):
            tmp = torch.index_select(m[i,:], 0, top_k_ind[i,:])
            d_re.append(tmp.tolist())
        print('d_re', d_re)


def flatten_itemFeat( item_feat_raw, tag_num, if_count):
      
       item_feats_vec = []
       for i  in range(len(item_feat_raw)):
           ifl = item_feat_raw[i] #'track_id', 'artist_ind', 'album_ind', 'tags'
           artist = ifl[1]
           if artist >= if_count[1]:
               raise AssertionError
           album = ifl[2]
           if album >= if_count[2]:
               raise AssertionError
           tag_l = ifl[3]
           cur_ffl = [artist, album]
           # binary cat vector 
           cat_l = [0 for _ in range(tag_num)]
           for catInd in tag_l:
              cat_l[catInd] = 1.0 / len(tag_l)
           cur_ffl.extend(cat_l)
           
           item_feats_vec.append(cur_ffl)
       return item_feats_vec
    











