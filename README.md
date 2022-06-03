# KoIR
Korean Information Retrieval

## Run
Training and inference with colbert and proposed retrieval model
```
bash examples.sh
```

## Experiments

### Data statistics
- NaverKin datasets (네이버 지식인)

| Train | Valid | Test |
|:----:|:----:|:----:|
| 115,828 | 14,473 | 14,479 |

### Data example
| Title | Query | Positive Document | Negative Document |
|------|--------|--------|--------|
|렌즈에 대한 질문이요|안녕하세요 중학생인데요 질문몇개만 하겠습니다1.부평에서 렌즈를 사려는데 렌즈하나에 얼마정도하죠?렌즈를 5000원에 파는곳이 있나요?2.값싼 렌즈를 사서 하루종일 쓰려는데 눈에 무리가 가진 않을까요?3.렌즈를 처음 사서 껴보는데 >많이 아픈가요?4.동네 가까운 안경점에도 5000원이하인 렌즈를 파나요?혹시 판다면 도수는 넣을수 있을까요?(시력이 마이너스인데>도 가능할까요?)5.값싼 렌즈를 사면 몇번정도 쓸수있나요?6.렌즈도 유통기간이 있나요?만약 있다면 어느정도인가요?빠른 답변 부탁드립니다~|5000원짜리렌즈와그보다싼렌즈는당연히잇어요 제발부탁인데요3개월되기전에쓰고버리고다시사고그러세요진짜나중에 후회하실꺼예요.....눈에당연히무리가지요맨날충열되잇겟죠...색소를눈에넣어같고다니는데|개인적으로 루디 변색고글 추천합니다 방탄렌즈에 자외선의 양에따라 검게 변하거든요 제가 써본것중에 최고인것같아요 가격은 조금 비싼편이지만 하나 장만하면 좋을것 같아요|

### Results
| Model                  | Hit@1 | Hit@10 | Hit@100 | MRR@10 |
|------------------------|:----:|:----:|:----:|:----:|
| BM25 | 6.42 | 12.42 | 20.91 | 8.14 |
| KoColBERT | 22.29 | 41.08 | 60.80 | 28.01 |
| Ours<sup>†</sup> | 26.98 | 48.22 | 68.16 | 33.41 |

## Question mark
colbert에 masked query 만들면 성능 떨어지던데?

## ToDo
- [X] Training Objects
- [X] CL
- [X] BM25
