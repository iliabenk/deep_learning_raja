import pandas as pd
import sqlite3
import numpy as np
from tqdm.auto import tqdm
import spacy
from functools import lru_cache
from collections import Counter, OrderedDict
import ast
import matplotlib.pyplot as plt
import os

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
def get_base_verbs(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc}

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


def compare_to_imsitu(imsitu_file, my_file):
    # imsitu_verbs = [get_base_verbs(v) for v in pd.read_csv(imsitu_file)['0'].to_list()]
    imsitu_verbs = set(pd.read_csv(imsitu_file)['0'].to_list())

    more_imsitu_verbs_options = []
    for v in imsitu_verbs:
        if not v.endswith('ing'):
            more_imsitu_verbs_options.append(v)
        else:
            more_imsitu_verbs_options.append(v)
            base_v = v[:-3]
            more_imsitu_verbs_options.append(base_v)
            more_imsitu_verbs_options.append(base_v + 'e')

            if base_v[-1] == 'y':
                more_imsitu_verbs_options.append(base_v[:-1] + 'ie')

            if base_v[-1] == base_v[-2]:
                more_imsitu_verbs_options.append(base_v[:-1])

    imsitu_verbs = more_imsitu_verbs_options
    my_verbs_df = pd.read_csv(my_file)

    if 'generated_caption' not in my_verbs_df.columns:
        my_verbs_df.rename(columns={'cap': 'generated_caption'}, inplace=True)

    my_verbs = [get_verbs(v) for v in my_verbs_df['generated_caption']]
    my_verbs = set().union(*my_verbs)

    # my_verbs_list = [s for e in my_verbs for s in e]
    my_verbs_list = list(my_verbs)

    verbs_that_exist_in_imsitu = {v: my_verbs_list.count(v) for v in imsitu_verbs if my_verbs_list.count(v) > 0}
    num_mutual_verbs = len(verbs_that_exist_in_imsitu)

    pd.DataFrame(list(verbs_that_exist_in_imsitu)).to_csv(f"{os.path.splitext(my_file)[0]}_intersection_verbs_with_imsitu.csv")

    fig, axes = plt.subplots(2)

    axes[0].bar(list(verbs_that_exist_in_imsitu.keys())[:num_mutual_verbs//2], list(verbs_that_exist_in_imsitu.values())[:num_mutual_verbs//2])
    plt.sca(axes[0])
    plt.xticks(rotation=90, ha='right')

    axes[1].bar(list(verbs_that_exist_in_imsitu.keys())[num_mutual_verbs//2:], list(verbs_that_exist_in_imsitu.values())[num_mutual_verbs//2:])
    plt.sca(axes[1])
    plt.xticks(rotation=90, ha='right')

    plt.tight_layout()

    plt.suptitle(f"Dataset verbs intersection with imsitu: {sum(verbs_that_exist_in_imsitu.values())} / {len(my_verbs_list)}\n"
                 f"Unique verbs that intersect: {len(verbs_that_exist_in_imsitu)} / {len(set(my_verbs_list))}")


    # counter = Counter(obj for x in df.objs for obj in x)
    #
    # df = pd.DataFrame({'verbs': verbs_that_exist_in_imsitu.keys(),
    #                    'count': verbs_that_exist_in_imsitu.values()})


    pass

def filter_csv(input_path, output_path, filter_type, imsitu_verbs_file=None):
    # filter_type = 'rare' / integer
    df = pd.read_csv(input_path)
    columns_to_keep = ['cap', 'verbs', 'objs', 'n_objs']

    df = df[df['n_objs'] > 0]
    df = df[df['n_objs'] <= 3]

    df = df[df.cap.str.endswith('.')]
    df.verbs = df.verbs.apply(eval)

    if imsitu_verbs_file is not None: # filter captions with verbs that are in imsitu
        imsitu_verbs_list = pd.read_csv(imsitu_verbs_file)['0'].to_list()

        df = df[df.verbs.apply(lambda x: not any(y in imsitu_verbs_list for y in x))]

    df = df[df.verbs.apply(lambda x: all(len(y) > 3 for y in x))] # Ignore short verbs due to spacy errors
    verbs_counts = pd.Series(x for row in df.verbs for x in row).value_counts()

    if filter_type == 'rare':
        S = verbs_counts.sum()

        rare_verbs = verbs_counts[verbs_counts.cumsum() > 0.85 * S].copy()

        df_rare = df[df.verbs.apply(lambda x: all(y in rare_verbs.index for y in x))].copy()
        df_rare.reset_index(drop=True, inplace=True)
        df_rare = df_rare[columns_to_keep]
    else:
        filter_val = int(filter_type)

        verbs_set = set(verbs_counts.index)
        df_rare = pd.DataFrame()
        for verb in verbs_set:
            df2 = df[df.verbs.apply(lambda x: verb in x)]
            df = df[df.verbs.apply(lambda x: verb not in x)]

            rand_ind = np.random.permutation(len(df2))

            df_rare = pd.concat([df_rare, df2.iloc[rand_ind[:filter_val]]])

    df_rare = df_rare.drop_duplicates(subset=['cap'])
    df_rare.to_csv(output_path)

if __name__ == "__main__":
    # df = generate_csv(input_db_path='/Users/iliabenkovitch/Documents/mocha/files/out_ilia_mistral.db',
    #                   db_type='sql',
    #              output_path="/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_122k.csv")

    filter_csv(input_path="/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_122k.csv",
               output_path="/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_122k_filtered_small_repeating_verbs.csv",
               imsitu_verbs_file="/Users/iliabenkovitch/Documents/mocha/files/imsitu_verbs_extended.csv",
               filter_type=15)

    # df = pd.read_csv("/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_filtered.csv")
    # generate_hist(df)

    # df = generate_csv(input_db_path='/Users/iliabenkovitch/Documents/mocha/files/verb_captions_gpt4.csv',
    #                   db_type="csv",
    #              output_path="/Users/iliabenkovitch/Documents/mocha/files/verb_captions_gpt4_with_counts.csv")
    #
    # compare_to_imsitu(imsitu_file="/Users/iliabenkovitch/Documents/mocha/files/imsitu_verbs_base_form.csv",
    #                   my_file="/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_122k_filtered.csv")

    df = pd.read_csv("/Users/iliabenkovitch/Documents/mocha/files/verbs_captions_mistral_with_counts_122k_filtered_small_repeating_verbs.csv")
    generate_hist(df)
    pass
