import csv
import math
import copy
import torch
import string
import torch.nn as nn
import torch.nn.functional as F


class SimIR(nn.Module):
    def __init__(self, args, tokenizer, backbone):
        super(SimIR, self).__init__()

        self.args = args
        self.tokenizer = tokenizer

        self.retrieval = backbone
        self.config = self.retrieval.config

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, inputs):
        Q = self.embedding(input_ids=inputs['query_input_ids'],
                           token_type_ids=inputs['query_token_type_ids'],
                           attention_mask=inputs['query_attention_mask'])
    
        PosDoc = self.embedding(input_ids=inputs['pos_doc_input_ids'],
                                token_type_ids=inputs['pos_doc_token_type_ids'],
                                attention_mask=inputs['pos_doc_attention_mask'])

        NegDoc = self.embedding(input_ids=inputs['neg_doc_input_ids'],
                                token_type_ids=inputs['neg_doc_token_type_ids'],
                                attention_mask=inputs['neg_doc_attention_mask'])
        
        return self.score(Q, PosDoc, NegDoc)

    def score(self, Q, PD, ND):

        positive_similarity = self.cos(Q.unsqueeze(1), PD.unsqueeze(0)) / self.args.temperature
        negative_similarity = self.cos(Q.unsqueeze(1), ND.unsqueeze(0)) / self.args.temperature
        cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to(self.args.device)

        labels = torch.arange(cosine_similarity.size(0)).long().to(self.args.device)
        
        return (cosine_similarity, labels)

    def embedding(self,
                  input_ids=None,
                  token_type_ids=None,
                  attention_mask=None):

        logits = self.retrieval(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)[1]
        return logits
