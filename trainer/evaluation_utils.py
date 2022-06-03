import re
import torch
import numpy as np
import torch.nn as nn

from apex import amp
from tqdm import tqdm

from parallelformers import parallelize

from trainer.models.simir import SimIR
from trainer.models.colbert import ColBERT
from transformers import AutoModel, AutoTokenizer

import warnings
import logging

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])
warnings.filterwarnings("ignore")


def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1))

def print_performance(metric, step):
    for key, value in metric.items():
        print(f"{key}: {value/(step + 1)}")

def cal_hit(args, yhat, y, k=10):
    if args.model == 'colbert':
        yhat = yhat.topk(k=k, dim=-1)[1].cpu()
    elif args.model == 'simir' or args.model == 'BM25':
        yhat = torch.tensor(np.argpartition(-yhat, range(k))[0:k])
        
    hits = (y == yhat).nonzero()

    if len(hits) == 0: return 0.
    else: return 1.

def cal_mrr(args, yhat, y, k=10):
    if args.model == 'colbert':
        yhat = yhat.topk(k=k, dim=-1)[1].cpu()
    elif args.model == 'simir' or args.model == 'BM25':
        yhat = torch.tensor(np.argpartition(-yhat, range(k))[0:k])
        
    hits = (y == yhat).nonzero()[:, -1] + 1
    
    mrr_score = torch.reciprocal(hits.float())
    
    if len(mrr_score) == 0:
        return 0.
    else:
        return mrr_score

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    q_token, q_token_idx = '[Q]', tokenizer.convert_tokens_to_ids('[unused0]')
    d_token, d_token_idx = '[D]', tokenizer.convert_tokens_to_ids('[unused1]')
    return tokenizer

def get_model(tokenizer, args):
    if args.model == 'colbert':
        model = ColBERT(args, tokenizer, AutoModel.from_pretrained(args.backbone))
    elif args.model == 'simir':
        model = SimIR(args, tokenizer, AutoModel.from_pretrained(args.backbone))

    vocab = tokenizer.get_vocab()
    model.retrieval.resize_token_embeddings(len(vocab))

    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    #parallelize(model, num_gpus=2, fp16=True, verbose='simple')
    #model = amp.initialize(model, opt_level=args.opt_level)

    return model.eval()

def get_data(test_file_name, db_file_name):
   with open(test_file_name, 'r', encoding='utf-8') as f:
       lines = f.readlines()
       title_list = []
       query_list = []
       response_idx = []
       for line in lines:
           line = line.split('\t')
           title_list.append(line[0])
           query_list.append(line[1])
           response_idx.append(int(line[2].strip()))

   with open(db_file_name, 'r', encoding='utf-8') as f:
       lines = f.readlines()
       db_list = []
       for line in lines:
           line = line.split('\t')
           db_list.append(line[0])

   print(f"\n\t== Query List: {len(query_list)} ==")
   print(f"\t== Response Index: {len(response_idx)} ==")
   print(f"\t== DB List: {len(db_list)} ==\n")
   assert len(response_idx) == len(query_list) == len(title_list)

   return title_list, query_list, response_idx, db_list

def query_embedding(args,
                    model=None,
                    inputs=None,
                    tokenizer=None):

    inputs = tokenizer(inputs[0],
                       inputs[1],
                       return_tensors="pt",
                       max_length=args.query_max_len,
                       pad_to_max_length="right")

    inputs['input_ids'][:,:1] = torch.tensor([tokenizer.convert_tokens_to_ids('[unused0]')])
    if args.model == 'colbert':
        logits = model.query_embedding(input_ids=inputs['input_ids'].to(args.device),
                                   token_type_ids=inputs['token_type_ids'].to(args.device),
                                   attention_mask=inputs['attention_mask'].to(args.device))
    elif args.model == 'simir':
        logits = model.embedding(input_ids=inputs['input_ids'].to(args.device),
                                   token_type_ids=inputs['token_type_ids'].to(args.device),
                                   attention_mask=inputs['attention_mask'].to(args.device))

    return logits

def doc_embedding(args,
                  model=None,
                  inputs=None,
                  tokenizer=None):
    if args.model == 'colbert':
        embeded_outputs = torch.ones(1, args.doc_max_len, args.coldim, device='cpu')
    elif args.model == 'simir':
        embeded_outputs = torch.ones(1, model.config.hidden_size, device='cpu')

    print(f"\n\t===Corpus Embedding===")
    for start_idx in tqdm(range(0, len(inputs), args.batch)):
        with torch.no_grad():
            cur_embedding = inputs[start_idx:start_idx+args.batch]
            cur_embedding = tokenizer(cur_embedding,
                                      truncation=True,
                                      return_tensors="pt",
                                      max_length=args.doc_max_len,
                                      pad_to_max_length="right")

            batch = cur_embedding['input_ids'].size(0)
            doc_positioning = torch.tensor([tokenizer.convert_tokens_to_ids('[unused1]')]).repeat(batch).unsqueeze(1)

            cur_embedding['input_ids'][:, :1] = doc_positioning

            if args.model == 'colbert':
                logits = model.doc_embedding(input_ids=cur_embedding['input_ids'].to(args.device),
                                         token_type_ids=cur_embedding['token_type_ids'].to(args.device),
                                         attention_mask=cur_embedding['attention_mask'].to(args.device))

            elif args.model == 'simir':
                logits = model.embedding(input_ids=cur_embedding['input_ids'].to(args.device),
                                         token_type_ids=cur_embedding['token_type_ids'].to(args.device),
                                         attention_mask=cur_embedding['attention_mask'].to(args.device))
            #if start_idx>10000:break
            embeded_outputs = torch.cat([embeded_outputs, logits.cpu()], dim=0)
            
    if args.model == 'colbert':
        embeded_outputs = embeded_outputs[1:, :, :]
    elif args.model == 'simir':
        embeded_outputs = embeded_outputs[1:, :]

    print(f"\t===Corpus Complete===")
    return embeded_outputs.to(args.device)

def bm25_score_idx(title,
                   query,
                   model):
    
    score = model.get_scores((title+" "+query).split(" "))
    return score

def scoring(args, 
            logits=None,
            corpus_embeddings=None, 
            metric=None,
            res_id=None):

    if args.model == 'colbert':
        scores = (logits @ corpus_embeddings.permute(0, 2, 1)).max(2).values.sum(1)

    elif args.model == 'simir':
        cos_scores = pytorch_cos_sim(logits, corpus_embeddings)[0]
        scores = cos_scores.cpu()
    
    elif args.model == 'BM25':
        scores = logits

    scores = [cal_hit(args, scores, res_id, k=1), cal_hit(args, scores, res_id, k=10),
              cal_hit(args, scores, res_id, k=100), cal_mrr(args, scores, res_id, k=10),
              cal_mrr(args, scores, res_id, k=100)]

    for step, (key, value) in enumerate(metric.items()): metric[key] += scores[step]
