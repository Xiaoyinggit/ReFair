import argparse
import configparser
import torch
import numpy as np 
import random
import data_utils
from torch.utils.data import DataLoader
import time
import evaluate
from model import pretrainRNNModel
from utils import *
from torch.utils.tensorboard import SummaryWriter

import os
os.environ['CUDA_LAUNCH_BLOCKING']= '1'





def main():
    parser = argparse.ArgumentParser('pretrain simulator.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_file', type=str, help = "config", default='config/rnn_config')
    parser.add_argument('--data_path',type=str, default='data/30music_2left_new.pkl')
    parser.add_argument('--log_dir', help='logging dir', type=str, default='./log')
    parser.add_argument('--save_path', help='save dir', type=str, default='./30music_rnnmodels/30music_pretrained_rnn_ckpt')
    



    args = parser.parse_args()
    print('main args: ', args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))

    train_batch_size = int(config['META']['train_batch_size'])


    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


    # load data
    train_data = data_utils.RNNdata(args.data_path, int(config['META']['max_seq_len']), d_type = 'train')
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

    test_data = data_utils.RNNdata(args.data_path, int(config['META']['max_seq_len']), d_type='test')
    test_loader = DataLoader(test_data, batch_size=int(config['META']['test_batch_size']), shuffle=False)


    


    # initialize model
    
    model = pretrainRNNModel(config['MODEL'], item_feats=train_data.data_dict['item_feats'], device=device, train_batch_size=train_batch_size)
    
    print('learning_rate: ', float(config['META']['learning_rate']))
    #optimizer 
    optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad == True],
            lr=float(config['META']['learning_rate']),
        )
    
 
    
    

    # logging path
    log_f = open(args.save_path+'/training.log', 'w')
    summary_dir = args.save_path + '/summary'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summarywriter = SummaryWriter(log_dir=summary_dir)
    num_epochs = int(config['META']['TRAIN_EPOCHS'])

    print('[main] start pretraining!')

    
    for epoch in range(num_epochs):
        model.train()

        start_time = time.time()
        cum_loss=0
        cum_loss_info= {'rating_mse':0, 'ns_mse': 0}
        debug_info =  {'r_logit_mean': 0, 'r_label_mean': 0,
                      'pos_r_pred_mean':0, 'neg_r_pred_mean': 0,
                     'leave_ns_pred_0':0,  'leave_ns_pred_1':0, 'stay_ns_pred_0': 0, 'stay_ns_pred_1': 0
                     }
        num_batch = 0
        for batch_samples in train_loader:
            # put to cuda

            model.zero_grad()
            padded_samples, valid_bs = do_padding(batch_samples, device=device, batch_size=train_batch_size)
            pred_l = model(padded_samples, valid_bs, summarywriter) # [rating_pred, ns_pred]
            loss, loss_info, d_info = model.cal_loss(pred_l, padded_samples[2], padded_samples[-1], summarywriter)
            cum_loss += loss 

            # cumulate different types of loss
            for lt, v in loss_info.items():
                cum_loss_info[lt] += v

            for dt, v in d_info.items():
                debug_info[dt] +=v
            num_batch +=1
            


            loss.backward()
            optimizer.step()
            model.global_step +=1 
            
        log_f.write('Epoch {} \t loss:{}, cost time: {:.2f}, {}, {}\n'\
                        .format(epoch, cum_loss/num_batch, time.time()-start_time, ','.join(['%s:%f'%(lt, v/num_batch) for lt, v in cum_loss_info.items()]), ','.join(['%s:%f'%(lt, v/num_batch) for lt, v in debug_info.items()])))
        print('Epoch {} \t loss:{}, cost time: {:.2f}, {}, {}'\
                        .format(epoch, cum_loss/num_batch, time.time()-start_time, ','.join(['%s:%f'%(lt, v/num_batch) for lt, v in cum_loss_info.items()]), ','.join(['%s:%f'%(lt, v/num_batch) for lt, v in debug_info.items()])))
        

            


        if epoch % int(config['META']['eval_step']) == 0:
            model.eval()

            metric_re = evaluate.evaluate(model, test_loader, device=device, batch_size=train_batch_size)
            
            print('[Evaluate Results] =========== ')
            for r, vl in metric_re.items():
                log_f.write('%s: %s\ '%(r, str(vl)))

            # save model
            model_path = args.save_path+'/epoch_%d'%epoch
            torch.save(model.state_dict(), model_path)

            


    
    print('End. Training!')
    summarywriter.close()














    






if __name__  == '__main__':
    main()