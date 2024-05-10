import pandas as pd
import argparse
from datasets import load_dataset

from utils import (
    extract_objects,
    extract_verbs_llm,
    load_llm_pipe, 
    get_answers
)

def flatten_data(df):
    caps_flat, objs_flat = [], []
    for cap, objs in zip(df.gt_caption, df.generated_objs):
        for obj in objs:
            caps_flat.append(cap)
            objs_flat.append(obj)
    return tuple(caps_flat), tuple(objs_flat)

def unflatten_responses(responses_flat, df):
    responses_unflat = []
    i=0
    for objs in df.generated_objs:
        cur_responses = []
        for obj in objs:
            cur_responses.append(responses_flat[i])
            i+=1
        responses_unflat.append(cur_responses)
    
    assert(len(responses_unflat) == len(df.generated_objs))
    return responses_unflat

def apply_ignore_words(responses_flat, objs_flat):
    ignore_words = ['painting', 'drawing', 'photo', 'picture', 'portrait', 'photograph']
    for i, obj in enumerate(objs_flat):
        if obj in ignore_words:
            responses_flat[i] = 'ignore'
    return responses_flat

def get_llm_responses(df, llm_pipe):

    caps_flat, objs_flat = flatten_data(df)
    responses_flat = get_answers(caps_flat, objs_flat, llm_pipe)
    responses_flat = apply_ignore_words(responses_flat, objs_flat)
    responses = unflatten_responses(responses_flat,df)

    pd.DataFrame({"gt_caption": df.gt_caption,
                  "generated_caption": df.generated_caption,
                  "generated_objs": df.generated_objs,
                  "llm_responses": responses}).to_csv("responses.csv")

    return responses

def get_och_score(llm_responses):
    responses = []
    [responses.extend(resp_per_cap) for resp_per_cap in llm_responses]
    data = pd.Series(responses).str.lower().str.strip()
    dv = data.value_counts()
    d = dv.to_dict()
    return d['no'] / (d['yes'] + d['no'])


def eval(args):
    print("Loading Dataset\n")
    # och_dataset = load_dataset("moranyanuka/OpenCHAIR", cache_dir=args.cache_dir)['test']
    och_dataset = load_dataset("iliabenk/verbs_2", cache_dir=args.cache_dir)['train']
    df = pd.read_csv(args.generations_file_path)
    df['gt_caption'] = och_dataset['text']

    word_conc = pd.read_excel(args.concreteness_dataset_path)[['Word','Conc.M']].set_index("Word").to_dict()['Conc.M']

    print("\nLoading LLM\n")
    llm_pipe = load_llm_pipe(args)

    print("\nExtracting Generated Objects (Per Image)\n")
    df['generated_objs'] = extract_objects(captions=df.generated_caption.tolist(), conc_df=word_conc, llm_pipe=llm_pipe)



    print("\nGetting LLM Responses\n")
    llm_responses = get_llm_responses(df, llm_pipe)
    OpenCHAIR_score = get_och_score(llm_responses)
    print("\nOpenCHAIR Score: \n")
    print(OpenCHAIR_score)
    



if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-ckpt", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--concreteness-dataset-path", type=str,
                        default="Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--cache-dir", type=str, default="cache_dir")
    parser.add_argument("--generations-file-path", type=str, default="out_700_ilia_verbs.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    eval(args)
