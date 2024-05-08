import pandas as pd
import sqlite3
import numpy as np
from tqdm.auto import tqdm
import spacy
from functools import lru_cache
from collections import Counter, OrderedDict
import ast
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_trf")

def is_concrete(noun, t=4.5):
    if noun in concreteness:
        return concreteness[noun] > t
    return False

@lru_cache(maxsize=None)
def get_verbs(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc if token.pos_ == 'VERB'}

@lru_cache(maxsize=None)
def cap2objs(cap):
    verbs = get_verbs(cap)
    return {v for v in verbs}

def generate_csv(input_db_path, db_type, output_path):
    tqdm.pandas()

    if db_type == "sql":
        with sqlite3.connect(input_db_path) as con:
            df = pd.read_sql('select * from data', con)
    elif db_type == "csv":
        df = pd.read_csv(input_db_path)
        df["cap"] = df["generated_caption"]
    else:
        assert False

    df['verbs'] = df.cap.progress_apply(get_verbs)
    df['objs'] = df.cap.progress_apply(cap2objs)
    df['n_objs'] = df.objs.str.len()

    df.to_csv(output_path)

def generate_hist(df):
    df['objs'] = df['objs'].apply(ast.literal_eval) # If DF was saved as csv, the sets are kept as strings
    
    counter = Counter(obj for x in df.objs for obj in x)
    counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    top_n = bottom_n = middle_n = 80

    top_verbs = list(counter.keys())[:top_n]
    bottom_verbs = list(counter.keys())[-bottom_n:]

    top_freq = list(counter.values())[:top_n]
    bottom_freq = list(counter.values())[-bottom_n:]
    
    mid_verbs = list(counter.keys())[(len(counter) // 2) : (len(counter) // 2) + middle_n]
    mid_freq = list(counter.values())[(len(counter) // 2) : (len(counter) // 2) + middle_n]
    
    fig, axes = plt.subplots(3)
    
    axes[0].bar(top_verbs, top_freq)
    axes[0].set_title(f'Top {top_n} Verbs Histogram')
    plt.sca(axes[0])  # Set the current axis to axes[0] for x-axis rotation
    plt.xticks(rotation=45, ha='right')
    
    axes[1].bar(mid_verbs, mid_freq)
    axes[1].set_title(f'Middle {middle_n} Verbs Histogram')
    plt.sca(axes[1])  # Set the current axis to axes[1] for x-axis rotation
    plt.xticks(rotation=45, ha='right')
    
    axes[2].bar(bottom_verbs, bottom_freq)
    axes[2].set_title(f'Bottom {top_n} Verbs Histogram')
    plt.sca(axes[2])  # Set the current axis to axes[1] for x-axis rotation
    plt.xticks(rotation=45, ha='right')
    
    plt.suptitle(f"Total of {len(counter)} Verbs")
    
    plt.tight_layout()
    
    # Sample captions
    # df = df[df.n_objs > 0].copy()
    # df['min_count'] = df.objs.apply(lambda x: min(counter[y] for y in x))
    # df = df.sort_values(by='min_count')
    pass
# len(set.union(*df.objs))
#
# df.sample(10)
#
# c = Counter(obj for x in df.objs for obj in x)
#
# df_ = df[df.n_objs > 0].copy()
# df_['min_count'] = df_.objs.apply(lambda x: min(c[y] for y in x))
# df_ = df_.sort_values(by='min_count')
#
# df_small = df_.head(len(df_) // 10)
# len(df_small)
#
# df_small.sample(10)
# df_small.cap.value_counts()

if __name__ == "__main__":
    # df = generate_csv(input_db_path='/Users/iliabenkovitch/Documents/deep_learning_raja'
    #                                 '/mocha_code_vm_copy_always_latest'
    #                            '/mocha_code/out.db',
    #              output_path="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/verbs_with_counts.csv")

    # df = generate_csv(input_db_path='/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl'
    #                                 '/captions_5k.csv',
    #                   db_type="csv",
    #              output_path="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files"
    #                          "/verbs_with_counts_5k_dataset"
    #                          ".csv")

    df = pd.read_csv("/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files"
                             "/verbs_with_counts_5k_dataset"
                             ".csv")
    generate_hist(df)
    pass
