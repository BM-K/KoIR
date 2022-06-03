import csv
import torch
import logging

from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

import transformers
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, mode='normal'):
        self.args = args
        self.mode = mode
        self.metric = metric

        self.tokenizer = tokenizer
        self.file_path = file_path

        self.query_input_ids = []
        self.query_attention_mask = []
        self.query_token_type_ids = []

        self.pos_doc_input_ids = []
        self.pos_doc_attention_mask = []
        self.pos_doc_token_type_ids = []

        self.neg_doc_input_ids = []
        self.neg_doc_attention_mask = []
        self.neg_doc_token_type_ids = []

        """
        KLUE/BERT
        [CLS] 2
        [PAD] 0
        [UNK] 1
        [Q] 31500
        [D] 31501
        """
        self.q_token, self.q_token_idx = '[Q]', self.tokenizer.convert_tokens_to_ids('[unused0]')
        self.d_token, self.d_token_idx = '[D]', self.tokenizer.convert_tokens_to_ids('[unused1]')
        
        self.init_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.mask_token = self.tokenizer.mask_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.mask_token_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def load_data(self, type):
        
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            logger.info('Data pre-processing')
            for step, line in enumerate(tqdm(lines)):
                self.data2tensor(line)

        assert len(self.query_input_ids) == \
               len(self.pos_doc_input_ids) == \
               len(self.neg_doc_input_ids)

    def data2tensor(self, line):
        title, query, positive_doc, negative_doc = line.split('\t')
        
        query_tokens = self.tokenizer(title,
                                      query,
                                      truncation=True,
                                      return_tensors="pt",
                                      max_length=self.args.query_max_len,
                                      pad_to_max_length="right")

        positive_doc_tokens = self.tokenizer(positive_doc,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.doc_max_len,
                                             pad_to_max_length="right")

        negative_doc_tokens = self.tokenizer(negative_doc,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.doc_max_len,
                                             pad_to_max_length="right")
        
        query_tokens['input_ids'][0][0] = self.q_token_idx
        positive_doc_tokens['input_ids'][0][0] = self.d_token_idx
        negative_doc_tokens['input_ids'][0][0] = self.d_token_idx
        
        self.query_input_ids.append(query_tokens['input_ids'].squeeze(0))
        self.query_attention_mask.append(query_tokens['attention_mask'].squeeze(0))
        self.query_token_type_ids.append(query_tokens['token_type_ids'].squeeze(0))

        self.pos_doc_input_ids.append(positive_doc_tokens['input_ids'].squeeze(0))
        self.pos_doc_attention_mask.append(positive_doc_tokens['attention_mask'].squeeze(0))
        self.pos_doc_token_type_ids.append(positive_doc_tokens['token_type_ids'].squeeze(0))

        self.neg_doc_input_ids.append(negative_doc_tokens['input_ids'].squeeze(0))
        self.neg_doc_attention_mask.append(negative_doc_tokens['attention_mask'].squeeze(0))
        self.neg_doc_token_type_ids.append(negative_doc_tokens['token_type_ids'].squeeze(0))

        return True

    def __getitem__(self, index):
        inputs = {'query_input_ids': self.query_input_ids[index].to(self.args.device),
                  'query_attention_mask': self.query_attention_mask[index].to(self.args.device),
                  'query_token_type_ids': self.query_token_type_ids[index].to(self.args.device),

                  'pos_doc_input_ids': self.pos_doc_input_ids[index].to(self.args.device),
                  'pos_doc_attention_mask': self.pos_doc_attention_mask[index].to(self.args.device),
                  'pos_doc_token_type_ids': self.pos_doc_token_type_ids[index].to(self.args.device),

                  'neg_doc_input_ids': self.neg_doc_input_ids[index].to(self.args.device),
                  'neg_doc_attention_mask': self.neg_doc_attention_mask[index].to(self.args.device),
                  'neg_doc_token_type_ids': self.neg_doc_token_type_ids[index].to(self.args.device)}

        return inputs

    def __len__(self):
        return len(self.query_input_ids)


def get_loader(args, metric, tokenizer):
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data

    train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
    valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)

    train_iter.load_data('train')
    valid_iter.load_data('valid')

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True)}

    return loader


if __name__ == '__main__':
    get_loader('test')
