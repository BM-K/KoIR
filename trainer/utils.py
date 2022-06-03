import os
import csv
import torch
import logging
import itertools

from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score


logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.step = 0
        self.args = args

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def cal_performance(self, yhat, y):
        if self.args.model != 'colbert':
            return torch.FloatTensor([1.]), torch.FloatTensor([1.])

        with torch.no_grad():
            y = y.cpu()
            yhat = yhat.max(dim=-1)[1].cpu()

            acc = (yhat == y).float().mean()
            f1 = f1_score(y, yhat, average='macro')
            
            return acc, f1

    def performance_check(self, cp):
        print(f'\n\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==\n')

        if self.args.model == 'colbert':
            print(f'\n\t==Train Acc: {cp["ta"]:.4f} | Valid Acc: {cp["va"]:.4f}==')
            print(f'\t==Train F1: {cp["tf"]:.4f} | Valid F1: {cp["vf"]:.4f}==\n')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def save_config(self, cp):
        config = "Config>>\n"
        for idx, (key, value) in enumerate(self.args.__dict__.items()):
            cur_kv = str(key) + ': ' + str(value) + '\n'
            config += cur_kv
        config += 'Epoch: ' + str(cp["ep"]) + '\t' + 'Valid loss: ' + str(cp['vl']) + '\n'

        with open(self.args.path_to_save+self.args.ckpt.split('.')[0]+'_config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            
            unwrapped_model = config['accelerator'].unwrap_model(config['model'])
            config['accelerator'].save(unwrapped_model.state_dict(), sorted_path)

            self.save_config(cp)
            print(f'\n\t## SAVE Valid Loss: {cp["vl"]:.4f} ##')

        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True

        self.performance_check(cp)
