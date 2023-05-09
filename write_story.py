from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torch
import numpy as np
import random as r
import pandas as pd
import wandb
import time
import torch.nn.functional as F

device = 'cuda:1'

def tokenize(tokenizer, prompts):
    return tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)['input_ids']

def main():
    model = BartForConditionalGeneration.from_pretrained('./saves/bart_generator_200000').to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

    prompt = "[WP] The people who killed your sister are at your mercy. Your husband is standing at the door in silence. What do you do next?"
    prompt_ids = tokenize(tokenizer, prompt).to(device)

    output_ids = model.generate(prompt_ids, temperature=0.88, num_return_sequences=20, do_sample=True, output_scores=True, return_dict_in_generate=True)

    output_text = [tokenizer.decode(g, clean_up_tokenization_spaces=True, skip_special_tokens=True) for g in output_ids['sequences']]

    print(output_text)


if(__name__ == '__main__'):
    main()
