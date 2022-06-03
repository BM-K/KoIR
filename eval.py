import argparse

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from trainer.evaluation_utils import *

def set_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='colbert')
    parser.add_argument('--backbone', type=str, default='klue/bert-base')
    parser.add_argument('--model_path', type=str, default='output/base.pt')
    parser.add_argument('--test_file_name', type=str, default='data/test_with_idx.tsv')
    parser.add_argument('--db_file_name', type=str, default='data/db.tsv')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--coldim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--doc_max_len', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--hit_k', type=int, default=10)
    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    args = parser.parse_args()
    return args

def evaluation():
    args = set_args()
    title_list, query_list, response_id, database = get_data(args.test_file_name,
                                                             args.db_file_name)
    
    if args.model == 'BM25':
        tokenized_corpus = [doc.split(" ") for doc in database]
        bm25 = BM25Okapi(tokenized_corpus)
        
    else:
        tokenizer = get_tokenizer(args)
        model = get_model(tokenizer, args)

        corpus_embeddings = doc_embedding(args,
                                          model=model,
                                          inputs=database,
                                          tokenizer=tokenizer)

    metric = {'hit_1': 0., 'hit_10': 0., 'hit_100': 0., 'mrr_10': 0., 'mrr_100': 0.}

    for iter_, (title, query, res_id) in tqdm(enumerate(zip(title_list, query_list, response_id))):
        with torch.no_grad():
            
            if args.model == 'BM25':
                logits = bm25_score_idx(title, query, bm25)
                
                scoring(args,
                        logits=logits,
                        metric=metric,
                        res_id=res_id)  
            else:
                logits = query_embedding(args,
                                         model=model,
                                         inputs=[title, query],
                                         tokenizer=tokenizer)
                scoring(args,
                        logits=logits,
                        corpus_embeddings=corpus_embeddings,
                        metric=metric,
                        res_id=res_id)
        
    print_performance(metric, iter_)
        
if __name__ == '__main__':
    evaluation()
