# 문맥을 고려한 한국어 텍스트 데이터 증강 (Korean Text Augmentation Considering Context, **K-TACC**)

## Overview
`Random Masking Insertion` : BERT based 모델을 활용하여, 의미상 자연스러운 토큰을 삽입하는 형식으로 문장 augmentation 수행

## Usage
### 설치
```
bash install.sh
```

### Pilot test
`augmentation.ipynb` 파일에서 Random Masking Insertion을 Data Augmentation 기법으로 택한 이유를 살펴볼 수 있습니다.

### 증강 데이터 생성
실행 전 데이터셋의 경로, 저장할 파일 이름을 직접 지정해야 합니다.
```
python augmentation.py
```

### 증강 예시
|증강 방법|원본 문장|증강 문장|
|---|---|---|
|BERT_Augmentation (Random Masking Insertion)|이순신은 매우 뛰어난 장군이다.|이순신은 매우 뛰어난 _조선의_ 장군이다.|


### STS 성능 평가
실행 전 wandb login이 필요합니다.
```
cd sts
bash train.sh
```

## Experiment

|Model|Pearson's correlation|
|---|---|
|base|0.9232|
|EDA (Random Deletion) | 0.8960|
|EDA (Random Swap) | 0.9243 |
|EDA (Random Synonym Replacement) | 0.9250 |
|EDA (Random Insertion) | 0.9259 |
|AEDA | 0.9252 |
|Adverb augmentation | 0.9299 |
|BERT_Augmentation (Random Masking Replacement) | 0.9023 |
|BERT_Augmentation (Random Masking Insertion) | **0.9300** |



## Reference
```
@article{karimi2021aeda,
  title={Aeda: An easier data augmentation technique for text classification},
  author={Karimi, Akbar and Rossi, Leonardo and Prati, Andrea},
  journal={arXiv preprint arXiv:2108.13230},
  year={2021}
}

@article{wei2019eda,
  title={Eda: Easy data augmentation techniques for boosting performance on text classification tasks},
  author={Wei, Jason and Zou, Kai},
  journal={arXiv preprint arXiv:1901.11196},
  year={2019}
}

https://github.com/catSirup/KorEDA
https://github.com/KLUE-benchmark/KLUE-baseline
https://github.com/kyle-bong/K-TACC
```
