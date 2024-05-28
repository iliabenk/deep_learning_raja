import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import re
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import json
import sqlite3, logging, traceback

# model_name = 'meta-llama/Llama-2-70b-chat-hf'
# model_name = 'meta-llama/Llama-2-13b-hf'#-chat-hf'
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'#-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token = "[PAD]"
# tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token="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",
    cache_dir="OpenCHAIR/cache_dir"
)

pipe = pipeline(
    "text-generation",
    # batch_size=16,
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    # max_new_tokens=100
)

df = pd.read_csv('verb_captions_gpt4_with_counts.csv').rename(
    columns={
        'generated_caption': 'cap',
        'verbs': 'n'
    
    }

)
df.n = df.n.apply(eval)
df['nn'] = df.n.str.len()

n = pd.Series(x for row in df.n for x in row).value_counts()

S = n.sum()

rare_n = n[n.cumsum() > 0.50 * S].copy()

df_rare = df[(df.nn > 0) & (df.n.apply(lambda x: all(y in rare_n.index for y in x)))].copy()

def get_prompt(K=10):
    subdf = df_rare.sample(K)
    out = ''
    for row in subdf.itertuples():
        # out += str(row.n) + ' => ' + row.cap + '\n'
        out += row.cap + '\n'
    return out



con = sqlite3.connect('out_ilia.db')
cur = con.cursor()
try:
    cur.execute(f'''
    CREATE TABLE data (
        cap TEXT
    )
    ''')
except Exception as e:
    logging.error(traceback.format_exc())

query = """
insert into data (cap)
values (?)
"""

pbar = tqdm()

while True:
    prompt = get_prompt()
    # print(prompt)
    out = pipe(prompt, max_new_tokens=50, temperature=0.8, num_return_sequences=1)
        #   eos_token_id=13)
    for x in out:
        text_list = x['generated_text'][len(prompt):].strip().split('\n')

        for text in text_list:
            data = (text,)
            cur.execute(query, data)
            con.commit()
            pbar.update()