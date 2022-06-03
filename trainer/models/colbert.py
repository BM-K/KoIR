import csv
import math
import copy
import torch
import string
import torch.nn as nn
import torch.nn.functional as F


class ColBERT(nn.Module):
    def __init__(self, args, tokenizer, backbone):
        super(ColBERT, self).__init__()

        self.args = args
        self.tokenizer = tokenizer

        self.retrieval = backbone
        self.config = self.retrieval.config

        self.skiplist = {w: True
                         for symbol in string.punctuation
                         for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.colinear = nn.Linear(self.config.hidden_size, self.args.coldim)

    def forward(self, inputs):
        Q = self.query_embedding(input_ids=inputs['query_input_ids'],
                                 token_type_ids=inputs['query_token_type_ids'],
                                 attention_mask=inputs['query_attention_mask'])
    
        PosDoc = self.doc_embedding(input_ids=inputs['pos_doc_input_ids'],
                                    token_type_ids=inputs['pos_doc_token_type_ids'],
                                    attention_mask=inputs['pos_doc_attention_mask'])

        NegDoc = self.doc_embedding(input_ids=inputs['neg_doc_input_ids'],
                                    token_type_ids=inputs['neg_doc_token_type_ids'],
                                    attention_mask=inputs['neg_doc_attention_mask'])

        return self.score(Q, PosDoc, NegDoc)

    def score(self, Q, PD, ND):
        positive_score = (Q @ PD.permute(0, 2, 1)).max(2).values.sum(1)
        negative_score = (Q @ ND.permute(0, 2, 1)).max(2).values.sum(1)

        score = torch.cat([positive_score.unsqueeze(1), negative_score.unsqueeze(1)], dim=-1)
        
        return score

    def query_embedding(self,
                        input_ids=None,
                        token_type_ids=None,
                        attention_mask=None):

        Q = self.retrieval(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)[0]
        
        return self.normalize(self.colinear(Q))

    def doc_embedding(self,
                      input_ids=None,
                      token_type_ids=None,
                      attention_mask=None):

        D = self.retrieval(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)[0]

        mask = torch.tensor(self.punctuation_mask(input_ids), device=self.args.device).unsqueeze(2).float()

        return self.normalize(self.colinear(D) * mask)

    def normalize(self, logits):
        return F.normalize(logits, p=2, dim=2)

    def punctuation_mask(self, input_ids):
        return [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
