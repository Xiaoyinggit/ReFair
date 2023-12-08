import torch
from torcheval.metrics import AUC, BinaryAccuracy, MulticlassAccuracy
from utils import *

def evaluate(model, test_data_loader, device, batch_size):

    r_metrics = {'r_acc': BinaryAccuracy(device=device)}
    ns_metrics = {'r_acc': MulticlassAccuracy(num_classes=model.nS, average='none', device=device)}
    with torch.no_grad():

       for test_sample in test_data_loader:
            padded_samples, valid_bs = do_padding(test_sample, device=device, batch_size=batch_size)
            pred = model(padded_samples, valid_bs) # [rating_pred, ns_pred]


            r_metrics, ns_metrics = model.cal_metric(pred, r_label = padded_samples[2], ns_label=padded_samples[-1], r_metrics =r_metrics, ns_metrics=ns_metrics)

       metric_re = {}
       print('[Evaluate Results] =======rating=== ')
       for rn, rm in r_metrics.items():
           metric_re[rn]=rm.compute().tolist()
           print('[rating_metric] ', rn, ': ', metric_re[rn])

       print('[Evaluate Results] =======next_state=== ')
       for nn, n_metric in ns_metrics.items():
           metric_re[nn] = n_metric.compute().tolist()
           print('[next_state_metric]', nn, ':', metric_re[nn])

       return metric_re


        

