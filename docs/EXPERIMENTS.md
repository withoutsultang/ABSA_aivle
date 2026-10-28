# 개인 실험 기록

리뷰 3,000개를 train 2,100 / validation 900으로 분리했습니다(random_state=100). BERT 11개 출력, sigmoid > 0.5 기준입니다.

| 설정 | 일반 파인튜닝 | LoRA |
|---|---:|---:|
| 학습률 | 2e-8 | 2e-4 |
| Epoch | 5 | 5 |
| Batch size | 32 | 32 |
| 학습 시간(초) | 911.5279 | 531.4986 |
| LoRA r / alpha / dropout | — | 8 / 32 / 0.1 |
| 적용 계층 | 전체 학습 | query, value |

수치는 원본 노트북에 저장된 당시 결과입니다. 학습률 등 설정이 다른 두 실행의 관찰값으로 해석합니다. 전체 11개 속성의 precision/recall/F1/support는 [CSV](../results/personal_per_aspect.csv)에, 설정은 [JSON](../results/experiment_settings.json)에 보관했습니다.

팀의 3단계 실습은 별도 notebooks/team에 구분했습니다. 팀 실험의 성과를 개인 실험 수치와 합산하지 않습니다.
