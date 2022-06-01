# KoIR
Korean Information Retrieval

## Experiments

### Data statistics
- NaverKin datasets (Low data)

| Train | Valid | Test |
|:----:|:----:|:----:|
| 115,828 | 14,473 | 14,479 |

### Results
| Model                  | Hit@1 | Hit@10 | Hit@100 | MRR@10 |
|------------------------|:----:|:----:|:----:|:----:|
| BM25 | 6.42 | 12.42 | 20.91 | 8.14 |
| KoColBERT | 22.08 | 40.38 | 59.90 | 27.53 |
| Ours<sup>†</sup> | 27.34 | 48.30 | 68.56 | 33.83 |

## ToDo
- [X] Training Objects
- [X] CL
- [X] BM25
