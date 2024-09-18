# Making Augset
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from tqdm import tqdm
from multiprocessing import Pool
import joblib
from functools import partial
import numpy as np
tqdm.pandas()

BERT_aug = BERT_Augmentation()
random_masking_insertion = BERT_aug.random_masking_insertion

orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')


# random insertion 
def apply_random_masking_insertion(x, ratio=0.15):
    return random_masking_insertion(x, ratio=ratio)

ratio = 0.15
random_masking_insertion_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer='threads')
mapper = joblib.delayed(apply_random_masking_insertion)
tasks = [mapper(row) for i, row in random_masking_insertion_train['sentence1'].items()]
random_masking_insertion_train['sentence1'] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in random_masking_insertion_train['sentence2'].items()]
random_masking_insertion_train['sentence2'] = pool(tqdm(tasks))

random_masking_insertion_augset = pd.concat([orig_train, random_masking_insertion_train])
random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_insertion_augset))
random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset_span_0.15_nospan.json')