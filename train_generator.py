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
def calc_ppl(model, x, y):
    with torch.no_grad():
        outputs = model(x, labels=y)
    probs = F.softmax(outputs['logits'], dim=2)

    right_probs = probs[:, :, y].squeeze()
    right_probs = [right_probs[i,i].item() for i in range(probs.shape[1])]

    log_probs = [np.log(x) for x in right_probs]
    log_sum = sum(log_probs)
    log_avg = 0 - (log_sum / len(log_probs))

    my_ppl = np.exp(log_avg)

    return my_ppl

def tokenize(tokenizer, prompts, model_type='BART'):
    if(model_type == 'BART'):
        return tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)['input_ids']

class StringDataset(Dataset):
    def __init__(self, prompts, stories):
        self.prompts = prompts
        self.stories = stories

    def __getitem__(self, index):
        return self.prompts[index], self.stories[index]

    def __len__(self):
        return len(self.prompts)

def get_loader(tokenizer, file_name, batch_size):
    print(f"loading '{file_name}'...")
    df = pd.read_csv(file_name)
    prompts, stories = list(df['prompts']), list(df['scores'])
    #prompts = tokenize(tokenizer, list(df['prompts']))
    #stories = tokenize(tokenizer, list(df['scores']))
    return DataLoader(StringDataset(prompts, stories), batch_size=batch_size, shuffle=False)

def test_ppl(test_loader, model):
    model.eval()
    print()
    ppls = []
    for i, (x, y) in enumerate(test_loader):
        x, y = tokenize(tokenizer, x), tokenize(tokenizer, y)
        x, y = x.to(device), y.to(device)

        loss, logits = model(x, labels=y)[:2]
        ppl = np.exp(loss.detach().cpu().numpy())
        print(f'\r{i}/{len(test_loader)}', end="")

        #my_ppl = calc_ppl(model, x, y)
        #ppls.append(my_ppl)
        #print("my_ppl", my_ppl)
        #print("ppl", ppl)
        #exit()
        ppls.append(ppl)

    return sum(ppls) / len(ppls)

def train_model(model, train_loader, val_loader, eval_every=10000, max_epochs=20):
    print('starting training...')

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    step, val_step = 0, 0
    t0 = time.perf_counter()
    for e in range(max_epochs):
        train_ppls = []
        for i, (x, y) in enumerate(train_loader):
            x, y = tokenize(tokenizer, x), tokenize(tokenizer, y)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss, logits = model(x, labels=y)[:2]
            train_ppls.append(np.exp(loss.item()))

            loss.backward()
            optimizer.step()

            t1 = time.perf_counter()
            time_per = ((e * len(train_loader)) + i + 1) / (t1 - t0)
            time_total = max_epochs * len(train_loader) / time_per

            step += 1
            wandb.log({"step":step, "loss":loss})
            print(f"\rtime:{t1 - t0:,.2f}/{time_total:,.2f} step:{step+1}/{max_epochs * len(train_loader):,} ppl:{sum(train_ppls[-20:]) / len(train_ppls[-20:]):.2f}------", end="")

            if(step % eval_every == 0):
                val_ppl = test_ppl(val_loader, model)
                val_step += 1
                wandb.log({"val_step":val_step, "val_ppl":val_ppl, "step":step, "loss":loss})
                print(f"\nval_num:{val_step} ppl:{val_ppl}\n")

                model.save_pretrained(f"./saves/bart_generator_{step}")

        print(f"\nepoch:{e} average loss:{sum(train_ppls)/len(train_ppls)}")


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)
def main():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

    wandb.init(project="story_generation", name='bart_generator')
    config = wandb.config
    wandb.watch(model)

    test_loader = get_loader(tokenizer, './test.csv', 5)
    val_loader = get_loader(tokenizer, './val.csv', 1)
    train_loader = get_loader(tokenizer, './train.csv', 5)

    train_model(model, train_loader, val_loader, 50000, 20)

if(__name__ == '__main__'):
    main()
