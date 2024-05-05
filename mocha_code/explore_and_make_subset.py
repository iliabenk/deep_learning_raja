import pandas as pd
import sqlite3
import numpy as np
from tqdm.auto import tqdm
import spacy
from functools import lru_cache
from collections import Counter

def is_concrete(noun, t=4.5):
    if noun in concreteness:
        return concreteness[noun] > t
    return False

@lru_cache(maxsize=None)
def get_nouns(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc if token.pos_ == 'NOUN'}

@lru_cache(maxsize=None)
def cap2objs(cap):
    nouns = get_nouns(cap)
    return {n for n in nouns if is_concrete(n)}

nlp = spacy.load("en_core_web_trf")

df_conc = pd.read_excel('OpenCHAIR/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx')
concreteness = df_conc.set_index('Word')['Conc.M'].to_dict()

tqdm.pandas()

with sqlite3.connect('out.db') as con:
    df = pd.read_sql('select * from data', con)

df = df.iloc[:1000] # vers1: Debug, do only on a 1k subset
df['nouns'] = df.cap.progress_apply(get_nouns)
df['objs'] = df.cap.progress_apply(cap2objs)
df['n_objs'] = df.objs.str.len()

len(set.union(*df.objs))

df.sample(10)

c = Counter(obj for x in df.objs for obj in x)

df_ = df[df.n_objs > 0].copy()
df_['min_count'] = df_.objs.apply(lambda x: min(c[y] for y in x))
df_ = df_.sort_values(by='min_count')

df_small = df_.head(len(df_) // 10)
len(df_small)

df_small.sample(10)
df_small.cap.value_counts()