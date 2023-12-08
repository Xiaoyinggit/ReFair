
import torch.utils.data as data
import pickle 
import numpy as np
import utils

class RNNdata(data.Dataset):

    def __init__(self, data_path, max_seq_len, d_type='train'):
        self.d_type = d_type
        self.max_seq_len = max_seq_len


        self.data_dict = self.loadData(data_path)
        self.geneSamples()


    def __len__(self):
       return len(self.user_feat_l)
    
    def __getitem__(self, idx):
       user_feats = self.user_feat_l[idx]
       item_id= self.itemID_l[idx]
       label = self.label_l[idx]
       hist_seq = self.hist_seq_l[idx]
       seq_len = self.seq_len_l[idx]
       ns = self.next_state_l[idx]
       return user_feats, item_id, label, hist_seq, seq_len, ns
    


    def geneSamples(self):
        # create for training samples
        self.user_feat_l = []
        self.itemID_l = []
        self.label_l = []
        self.hist_seq_l =[]
        self.seq_len_l = []
        self.next_state_l = []

        ns_stat = 0
        for dp in self.data_dict['%s_set'%self.d_type]:
           user_id, item, fb, hist_seq, next_state = dp 

           uf_list = self.data_dict['user_feat'][user_id]
           assert uf_list[0] == user_id
           
           # add to self.user_feats 
           self.user_feat_l.append(np.array(uf_list))
        
           self.itemID_l.append(item)
           
           
           self.label_l.append(fb)

           # padding for hist_seq 
           cur_seq_len = len(hist_seq)
           if len(hist_seq) < self.max_seq_len:
              padding = [[0,0] for _ in range(self.max_seq_len-len(hist_seq))]
              hist_seq.extend(padding)
           assert len(hist_seq) == self.max_seq_len
           
           self.hist_seq_l.append(np.array(hist_seq))
           self.seq_len_l.append(cur_seq_len)
           
           # origin ns: [0,1,2]
           new_ns = 0 #[0,1] for stay
           if next_state >=1:
              new_ns = 1 # 1 for leave
           ns_stat += new_ns
           self.next_state_l.append(new_ns)


        assert len(self.user_feat_l)==len(self.itemID_l)==len(self.label_l)

        print('[RNNdata] gene %d %s samples; ns_pos_ratio: %d (%f)'%(len(self.user_feat_l), self.d_type, ns_stat,ns_stat/len(self.user_feat_l)))



          
            



    def loadData(self, file_path):
       
       data_dict = {}
       with open(file_path, 'rb') as f:
         train_set = pickle.load(f)
         test_set = pickle.load(f)
         data_dict['user_count'], data_dict['item_count'] = pickle.load(f)
         data_dict['user_feat'], data_dict['uf_count'] = pickle.load(f)
         data_dict['item_feat_raw'], data_dict['if_count'] = pickle.load(f)
      
       if self.d_type == 'all':
           data_dict['%s_set'%self.d_type] = [*train_set, *test_set]
           print('train_set: ', len(train_set), 'test_set: ', len(test_set), 'all', len(data_dict['%s_set'%self.d_type]))

       else:
           data_dict['%s_set'%self.d_type] = train_set if self.d_type=='train' else test_set
       
           

       # flat item_feat
       print('item_feat: ', len(data_dict['item_feat_raw']),data_dict['item_count'] )
       assert len(data_dict['item_feat_raw']) == data_dict['item_count']

       data_dict['item_feats']= utils.flatten_itemFeat(data_dict['item_feat_raw'], data_dict['if_count'][-1],  data_dict['if_count'])
       assert len(data_dict['item_feats']) == data_dict['item_count']

     

       

       print('[loadData] load data for %d users,  %d items , %d interactions'%(data_dict['user_count'],  data_dict['item_count'], len(data_dict['%s_set'%self.d_type])))
       return data_dict
    
 
  
       

    def get_data_stat(self):
       # data statistic for intialize model
       data_stat = {}
       data_stat['uf_count']=  self.data_dict['uf_count']
       data_stat['user_count'] = self.data_dict['user_count']
       data_stat['item_count'] = self.data_dict['item_count']
       data_stat['item_cat'] = self.data_dict['if_count'][1]


       return data_stat


def load_item_feats(data_path):
      with open(data_path, 'rb') as f:
         train_set = pickle.load(f)
         test_set = pickle.load(f)
         user_count, item_count = pickle.load(f)
         user_feat, uf_count = pickle.load(f)
         item_feat_raw, if_count = pickle.load(f)
      
       # flat item_feat
      assert len(item_feat_raw) == item_count
      print('if_count: ', if_count)
      
      item_feats_vec = utils.flatten_itemFeat(item_feat_raw, if_count[-1], if_count)

   
      return item_feats_vec
       