#!/bin/bash
 
CUDA_VISIBLE_DEVICES=1 python main.py --model colbert --ckpt colbert.pt
CUDA_VISIBLE_DEVICES=1 python eval.py --model colbert --model_path output/colbert.pt 

CUDA_VISIBLE_DEVICES=1 python main.py --model simir --ckpt simir.pt
CUDA_VISIBLE_DEVICES=1 python eval.py --model simir --model_path output/simir.pt
