# KoIR
Korean Information Retrieval

## Run
Training and inferencing with colbert and proposed retrieval model
```
bash examples.sh
```

## Experiments

### Data statistics
- NaverKin datasets (네이버 )

| Train | Valid | Test |
|:----:|:----:|:----:|
| 115,828 | 14,473 | 14,479 |

### Results
| Model                  | Hit@1 | Hit@10 | Hit@100 | MRR@10 |
|------------------------|:----:|:----:|:----:|:----:|
| BM25 | 6.42 | 12.42 | 20.91 | 8.14 |
| KoColBERT | 22.29 | 41.08 | 60.80 | 28.01 |
| Ours<sup>†</sup> | 26.98 | 48.22 | 68.16 | 33.41 |

## Question mark
colbert에 mask query 만들면 성능 떨어지던데? 뭐가 잘못됐나

## ToDo
- [X] Training Objects
- [X] CL
- [X] BM25
