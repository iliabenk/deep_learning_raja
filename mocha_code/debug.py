from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

llm_ckpt = "openlm-research/open_llama_3b_v2"
tokenizer = AutoTokenizer.from_pretrained(llm_ckpt)
print()
