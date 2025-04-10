#!/usr/bin/env python3
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, AutoModel
from transformers import Gemma3ForConditionalGeneration
import os

# Get the Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN not set. Some models may require authentication.")

# Define the models to download
models = {
    #'llm_model': ('google/gemma-3-4b-it', Gemma3ForConditionalGeneration, AutoProcessor),
    'tts_model': ('ai4bharat/IndicF5', AutoModel, None),
    #'asr_model': ('ai4bharat/indic-conformer-600m-multilingual', AutoModel, None),
    'trans_en_indic': ('ai4bharat/indictrans2-en-indic-dist-200M', AutoModelForSeq2SeqLM, AutoTokenizer),
    'trans_indic_en': ('ai4bharat/indictrans2-indic-en-dist-200M', AutoModelForSeq2SeqLM, AutoTokenizer),
    'trans_indic_indic': ('ai4bharat/indictrans2-indic-indic-dist-320M', AutoModelForSeq2SeqLM, AutoTokenizer),
}

# Directory to save models
save_dir = '/app/models'

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Download and save each model
for name, (model_name, model_class, processor_class) in models.items():
    print(f'Downloading {model_name}...')
    model = model_class.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model.save_pretrained(f'{save_dir}/{name}')
    if processor_class:
        processor = processor_class.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        processor.save_pretrained(f'{save_dir}/{name}')
    print(f'Saved {model_name} to {save_dir}/{name}')