
from tqdm.auto import tqdm
from itertools import islice
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
from functools import lru_cache
import spacy
import en_core_web_sm, en_core_web_trf

import pandas as pd
import string

from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]
     

def is_concrete(noun, concretness, t=4.5):
    if noun in concretness:
        return concretness[noun] > t
    return False

def extract_objects(captions, conc_df=None, llm_pipe=None):
    # nouns = extract_nouns(captions, conc_df)
    verbs = extract_verbs_spacy(captions, llm_pipe)

    # objs = [n + v for n, v in zip(nouns, verbs)]
    objs = verbs

    return objs

def extract_nouns(captions, conc_df):
    nlp = en_core_web_sm.load()
    objs = []
    for caption in tqdm(captions):
        doc = nlp(caption.lower())
        cur_objs = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and is_concrete(token.lemma_, conc_df)]
        objs.append(cur_objs)
    return objs

def extract_verbs_spacy(captions, llm_pipe):
    def _make_prompt(sentence, verb, tokenizer):
        _prompt = f'''I will provide a sentence, and a word that exists in the sentence. 
        My question to you is whether this word should be tagged as a verb or a different part-of-speech in the sentence.
        Ignore gramatical issues or whether the described situtation is possible or not. Only whether it is a verb or not in the context of the provided sentence
        Provide an answer as follows: 
        Provide an explanation for your answer and whether you think the word is a noun / adjective / etc.
        Do it step by step, and finally provide the final answer as yes / no.
        Restrict yourself to 50 tokens-length explanation
        Here are few examples:
        sentence: A pile of dried seaweed is being used as fertilizer.
        word: dried
        explanation: dried is an adjective. Answer: no
        sentence: A group of people eating pizza in a pizzeria.
        word: eating
        explanation: the group is doing some action on the pizza. Answer: yes
        sentence: a stray dog is sniffing a piece of bacon.
        word: sniffing
        explanation: the dog is applying an action on the bacon. Answer: yes
        sentence: a dog surrounded by cats
        word: surrounded
        explanation: being surrounded by something is a verb. Answer: yes
        sentence: a dead horse drinking coffee
        word: drinking
        explanation: this is the action applied by the horse on the coffee. Answer: yes
        sentence: a mouse eating mozzarella
        word: cheese
        explanation: the action applied on the mozzarella by the mouse. Answer: yes
        sentence: A person in a full-body snowsuit is walking through a snowy area.
        word: walking
        explanation: this is the action the person does. Answer: yes
        sentence: A man's profile next to a car window with a sign saying "Never mind the gearbox, just get in the car".
        word: saying
        explanation: this is a phrase, a sign does not actually say that. Answer: no
        sentence: a tiger eating a zebra
        word: eating
        explanation: the action the tiger does. Answer: yes
        sentence: sliced beets on a cutting board.
        word: sliced
        explanation: the word "sliced" functions as an adjective describing the state of the beets. Answer: no
        sentence: {sentence}
        word: {verb}
        explanation: '''

        prompt = tokenizer.apply_chat_template([{'role':'user', "content":_prompt}], tokenize=False)
        return prompt
    
    def _make_second_prompt(sentence, verb, llm_response, tokenizer):
        _prompt = f'''I will provide a sentence, and a word that exists in the sentence. 
        My question to you is whether this word should be tagged as a verb or a different part-of-speech in the sentence.
        Ignore gramatical issues or whether the described situtation is possible or not. Only whether it is a verb or not in the context of the provided sentence
        I will provide an explanation and based on that you provide your answer.
        Only provide an answer with one word: yes / no
        Here are few examples:
        sentence: A pile of dried seaweed is being used as fertilizer.
        word: dried
        explanation: dried is an adjective. Answer: no
        sentence: A group of people eating pizza in a pizzeria.
        word: eating
        explanation: the group is doing some action on the pizza. Answer: yes
        sentence: a stray dog is sniffing a piece of bacon.
        word: sniffing
        explanation: the dog is applying an action on the bacon. Answer: yes
        sentence: a dog surrounded by cats
        word: surrounded
        explanation: being surrounded by something is a verb. Answer: yes
        sentence: a dead horse drinking coffee
        word: drinking
        explanation: this is the action applied by the horse on the coffee. Answer: yes
        sentence: a mouse eating mozzarella
        word: cheese
        explanation: the action applied on the mozzarella by the mouse. Answer: yes
        sentence: A person in a full-body snowsuit is walking through a snowy area.
        word: walking
        explanation: this is the action the person does. Answer: yes
        sentence: A man's profile next to a car window with a sign saying "Never mind the gearbox, just get in the car".
        word: saying
        explanation: this is a phrase, a sign does not actually say that. Answer: no
        sentence: a tiger eating a zebra
        word: eating
        explanation: the action the tiger does. Answer: yes
        sentence: sliced beets on a cutting board.
        word: sliced
        explanation: the word "sliced" functions as an adjective describing the state of the beets. Answer: no
        sentence: {sentence}
        word: {verb}
        explanation: {llm_response} Answer: '''

        prompt = tokenizer.apply_chat_template([{'role':'user', "content":_prompt}], tokenize=False)
        return prompt
    
    def _prompt_llm(pipe, prompt):
        output = pipe(prompt, max_new_tokens=150, do_sample=False, num_return_sequences=1)
        
        # print(output[0]['generated_text']) # debug
        # output = output[0]['generated_text'][len(prompt):].strip().translate(str.maketrans('', '', string.punctuation)).lower().split()
        # output = output[0]['generated_text'][len(prompt):].strip().lower().split()

        return output[0]['generated_text']

    df_responses = pd.DataFrame()
    
    nlp = en_core_web_trf.load()
    verbs = []
    for caption in tqdm(captions):
        doc = nlp(caption.lower())
        # cur_verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        _cur_verbs = [token for token in doc if token.pos_ == 'VERB']

        if len(_cur_verbs) == 0:
            verbs.append(_cur_verbs)
        else:
            cur_verbs = []

            for _verb in _cur_verbs:
                prompt = _make_prompt(caption.lower(), _verb, tokenizer=llm_pipe.tokenizer)
                llm_response = _prompt_llm(llm_pipe, prompt)

                df_responses = pd.concat([df_responses, pd.DataFrame({"generated_caption": [caption],
                                    "response": [llm_response[len(prompt):]]})])

                if "Answer:" not in llm_response[len(prompt):]:
                    prompt = _make_second_prompt(caption.lower(), _verb, llm_response[len(prompt):], tokenizer=llm_pipe.tokenizer)
                    llm_response = _prompt_llm(llm_pipe, prompt)

                    df_responses = pd.concat([df_responses, pd.DataFrame({"generated_caption": [caption],
                                        "response": [llm_response[len(prompt):]]})])

                llm_response_words_list = llm_response[len(prompt):].strip().translate(str.maketrans('', '', string.punctuation)).lower().split()
                
                if 'yes' in llm_response_words_list:
                    cur_verbs.append(_verb.lemma_)

            verbs.append(cur_verbs)

    df_responses.to_csv("verbs_llm_responses.csv")

    return verbs

def extract_verbs_llm(captions, llm_pipe):
    END_SEQ_TOKENS = "!@#"

    def _make_prompt(cap, tokenizer):
        _prompt = _prompt = f'''I will provide a sentence, write to me ONLY the verbs in their basic form that appear in the sentence, do not provide any explanation. If no verbs exist, return []. Finish your response with the sequence {END_SEQ_TOKENS}\n
          Here are few examples:\n
            Q: a face with a red makeup\n
            A: []{END_SEQ_TOKENS}\n
            Q: kids playing soccer\n
            A: play{END_SEQ_TOKENS}\n
            Q: a couple hugging under a tree\n
            A: hug{END_SEQ_TOKENS}\n
            Q: a child sitting at a desk writing a letter\n
            A: sit, write{END_SEQ_TOKENS}\n
            Q: {cap}\n
            A:'''
        prompt = tokenizer.apply_chat_template([{'role':'user', "content":_prompt}], tokenize=False)
        return prompt

    def _get_answers(captions, pipe):
        import string
        _punctuations = string.punctuation
        prompts = [_make_prompt(cap, pipe.tokenizer) for cap in captions]
        dataset = ListDataset(prompts)
        
        outputs = []
        with tqdm(total=len(prompts)) as pbar:
            for out in pipe(dataset, max_new_tokens=50, do_sample=False, num_return_sequences=1):
                outputs.append(out[0]["generated_text"])
                pbar.update(1)
        
        # outputs = [f"[{outputs[i][len(prompts[i]):].split(END_SEQ_TOKENS)[0].strip().translate(str.maketrans('', '', string.punctuation))}]" for i in range(len(outputs))]
        outputs = [outputs[i][len(prompts[i]):].split(END_SEQ_TOKENS)[0].strip() for i in range(len(outputs))]
        return outputs
    
    outputs = _get_answers(captions, llm_pipe)
    # nlp = en_core_web_sm.load()
    # nlp = en_core_web_lg.load()
    # nlp = en_core_web_md.load()
    verbs = []
    for caption in tqdm(captions):
        doc = nlp(caption.lower())
        cur_verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB' and is_concrete(token.lemma_, conc_df,
                                                                                           t=3.5)]
        verbs.append(cur_verbs)
    return verbs

def load_llm_pipe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckpt, token="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")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True)
    
    model = AutoModelForCausalLM.from_pretrained(args.llm_ckpt,
                                                 quantization_config=bnb_config,device_map="auto",
                                                 cache_dir=args.cache_dir,
                                                 token="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")
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                    device_map="auto",
                    batch_size=args.batch_size)
    return pipe

def parse_ans(ans):
    ans_word_list = ans.lower().replace(',','').replace('.','').replace(';','').replace('\n',' ').split(' ')
    if 'yes' in ans_word_list:
        return 'yes'
    elif 'no' in ans_word_list or 'not' in ans_word_list:
        return 'no'
    elif 'unsure' in ans_word_list:
        return 'unsure'
    else:
        return 'ERROR: '+';'.join(ans_word_list)

def make_prompt(cap, obj, tokenizer):
    # _prompt = f'''Here are a few descriptions of an image: {cap}.\nDoes the image contain the following object: {obj}?\nAnswer yes/no/unsure.\n The answer is: '''
    _prompt = f'''I will provide a description of an image and an object that could be a noun or verb, and your task is to say whether this object exists in the general sense in the description.
    When I say in the general sense I mean that don't check it grammatically, but conceptually. Answer with only one word: yes/no/unsure
    Here are few examples:
    description: a couple chatting
    object: talk
    A: yes
    description: a dog barking on another dog
    object: play
    A: no
    description: a mouse eating cheese
    object: cheese
    A: yes
    description: a mouse eating mozzarella
    object: cheese
    A: yes
    description: an owl sleeping on a tree
    object: sleep
    A: yes
    description: a cat playing in the snow
    object: rain
    A: no
    description: a tiger eating a zebra
    object: horse
    A: no
    description: {cap}
    object: {obj}
    A:'''
    prompt = tokenizer.apply_chat_template([{'role':'user', "content":_prompt}], tokenize=False)
    return prompt

@lru_cache(maxsize=None)
def get_answer(cap, obj, pipe):
    prompt = make_prompt(cap, obj, pipe.tokenizer)
    out = pipe(prompt, max_new_tokens=8, do_sample=False, num_return_sequences=1)
    out = out[0]['generated_text'][len(prompt):].strip()
    out = parse_ans(out)
    return out

@lru_cache(maxsize=None)
def get_answers(caps_flat, objs_flat, pipe):
    prompts = [make_prompt(cap, obj, pipe.tokenizer) for cap,obj in zip(caps_flat, objs_flat)]
    dataset = ListDataset(prompts)
    
    outputs = []
    with tqdm(total=len(prompts)) as pbar:
        for out in pipe(dataset, max_new_tokens=8, do_sample=False, num_return_sequences=1):
            outputs.append(out)
            pbar.update(1)
    
    outputs = [outputs[i][0]['generated_text'][len(prompts[i]):].strip() for i in range(len(outputs))]
    outputs = [parse_ans(out) for out in outputs]
    return outputs