import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
from trainer.loss import Loss
from trainer.utils import Metric
from accelerate import Accelerator
from data.dataloader import get_loader
from trainer.models.simir import SimIR
from trainer.models.colbert import ColBERT
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.hypo = []

        self.args = args
        self.config = None

        self.loss_fn = Loss(args)
        self.metric = Metric(args)

        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': 0, 'iter': 0, 'acc': 0, 'f1': 0}

    def run(self, inputs):

        score = self.config['model'](inputs)
        
        loss, labels = self.loss_fn.base(self.config, score)
        acc, f1 = self.metric.cal_performance(score, labels)

        return loss, acc, f1

    def progress(self, loss, acc, f1):
        self.model_progress['iter'] += 1
        self.model_progress['loss'] += loss
        self.model_progress['acc'] += acc
        self.model_progress['f1'] += f1

    def return_value(self):
        loss = self.model_progress['loss'].cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'] / self.model_progress['iter']
        f1 = self.model_progress['f1'] / self.model_progress['iter']

        return loss, acc, f1

    def get_object(self, tokenizer, model):
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.args.lr)

        return optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.args.warmup_ratio * train_total,
            num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        accelerator = Accelerator(fp16=True)

        tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)

        if self.args.model == 'colbert':
            model = ColBERT(self.args, tokenizer, AutoModel.from_pretrained(self.args.backbone))
        elif self.args.model == 'simir':
            model = SimIR(self.args, tokenizer, AutoModel.from_pretrained(self.args.backbone))
        else:
            model = None
        
        vocab = tokenizer.get_vocab()

        model.retrieval.resize_token_embeddings(len(vocab))
        
        if self.args.multi_gpu == 'True':
            model = nn.DataParallel(model, output_device=1)

        model.to(self.args.device)

        loader = get_loader(self.args, self.metric, tokenizer)

        optimizer = self.get_object(tokenizer, model)

        if self.args.test == 'False':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'accelerator': accelerator,
                  'args': self.args,
                  'model': model}
        """
        if config['args'].fp16 == 'True' and config['args'].test == 'False':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)
        """
        config['model'], config['optimizer'] = accelerator.prepare(model, optimizer)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)
        
        train_loader = self.config['accelerator'].prepare(self.config['loader']['train'])
        for step, batch in enumerate(tqdm(train_loader)):

            self.config['optimizer'].zero_grad()

            inputs = batch
            loss, acc, f1 = self.run(inputs)

            """
            if self.args.fp16 == 'True':
                with amp.scale_loss(loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            """
            loss = torch.mean(loss)
            acc = torch.mean(acc)
            
            if self.args.multi_gpu == 'True':
                f1 = torch.mean(f1)

            self.config['accelerator'].backward(loss)

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            
            self.progress(loss.data, acc.data, f1)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)
        
        valid_loader = self.config['accelerator'].prepare(self.config['loader']['valid'])
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):#self.config['loader']['valid']):

                inputs = batch
                loss, acc, f1 = self.run(inputs)
                loss = torch.mean(loss)
                acc = torch.mean(acc)
                
                if self.args.multi_gpu == 'True':
                    f1 = torch.mean(f1)

                self.progress(loss.data, acc.data, f1)

        return self.return_value()
