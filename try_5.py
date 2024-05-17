|%%--%%| <ajvEl0eORu|lCrdzqv4ot>

import torch
from utils import  JSONDataset, RevdictModel, PAD, EOS, BOS, UNK, AraT5RevDict
import tqdm # progree bar


#|%%--%%| <lCrdzqv4ot|gHn2qwYPcp>

# Step 1: Prepare the Dataset
dataset_file = 'dev.json'  # Replace with your dataset file path
dataset = JSONDataset(dataset_file)
#|%%--%%| <gHn2qwYPcp|Zis7H5ioFE>

print(dataset.tensors)

#|%%--%%| <Zis7H5ioFE|P4KvGPgiky>
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - 100, 100])
# print(dataset.vocab)
# Step 2: Model Selection
model = RevdictModel(dataset.vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=256)

#|%%--%%| <P4KvGPgiky|tX5iXzQ4mF>
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The RevdictModel model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

#|%%--%%| <tX5iXzQ4mF|2AusxkiuAX>

# Hayper
EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 10, 1.0e-4, 0.9, 0.999, 1.0e-6
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
)
loss_fn = torch.nn.MSELoss()
#|%%--%%| <2AusxkiuAX|sHZFz8H6D8>

# Step 3: Model Training
for epoch in tqdm.trange(EPOCHS, desc="Epoch"):  
    model.train()
    for batch in train_set:

        gloss_tensor = batch['gloss_tensor'].unsqueeze(1)
        gloss_emb = batch['electra_tensor'].unsqueeze(1)  # Choose one of the embeddings (bertseg, bertmsa, electra)
        # print(gloss_emb.size(), gloss_tensor.size())
        # break
        optimizer.zero_grad()
        output_emb = model(gloss_tensor)
        loss = loss_fn(output_emb, gloss_emb)
        loss.backward()
        optimizer.step()


#|%%--%%| <sHZFz8H6D8|WTrPutRJhI>
# Step 4: Model Evaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in test_set:
        gloss_tensor = batch['gloss_tensor'].unsqueeze(1)
        gloss_emb = batch['electra_tensor'].unsqueeze(1)
        # print(gloss_tensor)
        output_emb = model(gloss_tensor)
        loss = loss_fn(output_emb, gloss_emb)
        total_loss += loss.item()

avg_loss = total_loss / len(test_set)
print(f"Average Test Loss: {avg_loss:.4f}")

#|%%--%%| <WTrPutRJhI|7E3jnBnbkj>
# embedding_list=[data["electra_tensor"] for data in dataset]
# print(embedding_list[0])
#|%%--%%| <7E3jnBnbkj|PBHwitAMbT>


# Assuming 'model' is your trained model
model.vocab = dataset.vocab
# dataset.tensors = embedding_list

freeze_vocab=0
maxlen = 256
# Step 5: Reverse Dictionary Lookup
def find_similar_words(gloss, n=5):
    with torch.no_grad():
        # if gloss not in model.vocab:
        #     print(f"Gloss '{gloss}' is not in the vocabulary.")
            # return []
        # gloss_tensor = torch.tensor(model.vocab[gloss])
        # gloss_tensor = torch.clamp(gloss_tensor, min=0, max=model.embedding.num_embeddings - 1)
        # gloss_tensor = model.embedding(gloss_tensor)
        gloss_tensor= torch.tensor(
                [model.vocab[PAD]] + [
                    model.vocab[word]
                    if not freeze_vocab
                    else dataset.vocab.get(word, model.vocab[UNK])
                    for word in gloss.split()
                ]
                + [model.vocab[EOS]])

        if maxlen:
            gloss_tensor = gloss_tensor[:maxlen]

        gloss_emb = model(gloss_tensor.unsqueeze(1))
        distances = torch.cdist(gloss_emb, dataset.tensors, p=2)
        closest_indices = torch.topk(distances, n, largest=False).indices
        closest_words = [dataset.itos[idx] for idx in closest_indices.squeeze()]

        # print("Size of the vocabulary:", len(model.vocab))
        # print("Index being passed to embedding layer:", model.vocab[gloss])
        return closest_words

query_gloss = "غَمَّ وَأَمْرَضَ القَلْبَ"  # Replace with your query gloss
# query_gloss = dev_df.loc[0,"gloss"]
similar_words = find_similar_words(query_gloss)
if similar_words:
    print(f"Similar words for '{query_gloss}': {similar_words}")
else:
    print(f"No similar words found for '{query_gloss}'.")

#|%%--%%| <PBHwitAMbT|6KgAJ7b47Q>
import pandas as pd
dev_df=pd.read_json("dev.json")
print(dev_df.info())
print(dev_df.loc[1,"gloss"])

#|%%--%%| <6KgAJ7b47Q|6DlzWGpHqA>

# Assuming 'model' is your trained model
model.vocab = dataset.vocab
dataset.tensors = embedding_list

freeze_vocab=0
maxlen = 256
# Step 5: Reverse Dictionary Lookup
def find_similar_words(gloss, n=5):
    with torch.no_grad():
        # if gloss not in model.vocab:
        #     print(f"Gloss '{gloss}' is not in the vocabulary.")
            # return []
        # gloss_tensor = torch.tensor(model.vocab[gloss])
        # gloss_tensor = torch.clamp(gloss_tensor, min=0, max=model.embedding.num_embeddings - 1)
        # gloss_tensor = model.embedding(gloss_tensor)
        gloss_tensor= torch.tensor(
                [model.vocab[PAD]] + [
                    model.vocab[word]
                    if not freeze_vocab
                    else dataset.vocab.get(word, model.vocab[UNK])
                    for word in gloss.split()
                ]
                + [model.vocab[EOS]])

        if maxlen:
            gloss_tensor = gloss_tensor[:maxlen]

        gloss_emb = model(gloss_tensor.unsqueeze(1))
        dataset_tensor = dataset.tensors[0].unsqueeze(1)
        distances = torch.cdist(gloss_emb, dataset_tensor, p=2)
        closest_indices = torch.topk(distances, n, largest=False).indices
        closest_words = [dataset.itos[idx] for idx in closest_indices.squeeze()]

        # print("Size of the vocabulary:", len(model.vocab))
        # print("Index being passed to embedding layer:", model.vocab[gloss])
        return closest_words

# query_gloss = "غَمَّ وَأَمْرَضَ القَلْبَ"  # Replace with your query gloss
query_gloss = dev_df.loc[1,"gloss"]
similar_words = find_similar_words(query_gloss)
if similar_words:
    print(f"Similar words for '{query_gloss}': {similar_words}")
else:
    print(f"No similar words found for '{query_gloss}'.")
#|%%--%%| <6DlzWGpHqA|TmPwvv6WKI>


# [*map(lambda x:x[0], embeddings_df.values)]

#|%%--%%| <TmPwvv6WKI|IY2v6WXWVi>

import pandas as pd
dev_df = pd.read_json("dev.json")

print(dev_df.info())
print(dev_df.set_index("word"))
# print(dev_df.loc[0,["word", "gloss"]])
# dev_df.loc[0,"electra"]

#|%%--%%| <IY2v6WXWVi|2gfWkKiHD0>


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the word embeddings (assuming they are in a DataFrame)
# embeddings_df = pd.read_json("dev.json")
embeddings_df = pd.read_json("train.json")

# print(embeddings_df.columns)
# Drop unnecessary columns and set the word as the index
embeddings_df = embeddings_df.set_index("word").drop(["id", "pos", "gloss", "bertseg", "bertmsa"], axis=1)

# print(embeddings_df.dtypes)
# print(embeddings_df.iloc[0,0])
# print()
x = [*map(lambda x:x[0], embeddings_df.values)]
len(x)


#|%%--%%| <2gfWkKiHD0|gToCMHqVWi>

# # Perform PCA to reduce dimensionality to 2 dimensions
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform([*map(lambda x:x[0], embeddings_df.values)])

# # Create a new DataFrame with the reduced embeddings
reduced_df = pd.DataFrame(reduced_embeddings, index=embeddings_df.index, columns=["x", "y"])

# Plot the word embeddings using a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced_df["x"], reduced_df["y"])

# # Add labels for the words (you can adjust the font size as needed)
for word, pos in reduced_df.iterrows():
    plt.text(pos.x, pos.y, word, fontsize=8)

 # plt.xlabel("PCA Dimension 1")
 # plt.ylabel("PCA Dimension 2")
 # plt.title("Word Embedding Visualization using PCA")
 # plt.grid(True)
 # plt.show()


#|%%--%%| <gToCMHqVWi|oHtTw2IvrO>
r"""°°°
# revdict5
°°°"""
#|%%--%%| <oHtTw2IvrO|Qzfs6cOoQB>
import argparse
import itertools
import json
import logging
import pathlib
import sys

from utils import  JSONDataset, RevdictModel, PAD, EOS, BOS, UNK, AraT5RevDict

logger = logging.getLogger(pathlib.Path("x.log").name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

import tqdm

# import data
# import models

from transformers import AutoTokenizer
# from ard_dataset import 

#|%%--%%| <Qzfs6cOoQB|8GYYxcnxec>
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout)


class ARDDataset(Dataset):
    def __init__(self, path, is_test=False) -> None:
        super().__init__()
        self.is_test = is_test
        self.data = read_json(path)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.is_test:
            return sample["id"], sample["word"], sample["gloss"],
        else:
            return sample["id"], sample["word"], sample["gloss"], sample["electra"], sample["bertseg"], sample['bertmsa']

    def __len__(self):
        return len(self.data)
#|%%--%%| <8GYYxcnxec|gZW4QdCGmK>
def rank_cosine(preds, targets):
    assocs = F.normalize(preds) @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float()
    assert ranks.numel() == preds.size(0)
    ranks = ranks.mean().item()
    return ranks / preds.size(0)



#|%%--%%| <gZW4QdCGmK|kCSTM0MjE8>
def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    ## make datasets
    train_dataset = ARDDataset(args.train_file)
    valid_dataset = ARDDataset(args.dev_file)
    
    ## make dataloader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    ## make summary writer
    summary_writer = SummaryWriter(args.save_dir / args.summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    logger.debug("Setting up training environment")

    model = AraT5RevDict(args).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5v2-base-1024")     
    model.train()

    # 3. declare optimizer & loss_fn
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 20, 1.0e-4, 0.9, 0.999, 1.0e-6
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    vec_tensor_key = f"{args.target_arch}_tensor"

    best_cosine = 0

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for ids, word, gloss, electra, bertseg, bertmsa in train_dataloader:
            optimizer.zero_grad()

            word_tokens = tokenizer(word, padding=True, return_tensors='pt').to(args.device)
            gloss_tokens = tokenizer(gloss, padding=True, return_tensors='pt').to(args.device)

            if args.target_arch == "electra":
                target_embs = torch.stack(electra, dim=1).to(args.device)
            elif args.target_arch =="bertseg":
                target_embs = torch.stack(bertseg, dim=1).to(args.device)
            elif args.target_arch =="bertmsa":
                target_embs = torch.stack(bertmsa, dim=1).to(args.device)


            target_embs = target_embs.float()
            ce_loss, pred_embs = model(
                gloss_tokens["input_ids"], 
                gloss_tokens["attention_mask"],
                word_tokens["input_ids"],
            )

            mse_loss = loss_fn(pred_embs, target_embs)
            loss = args.ce_loss_weight * ce_loss + mse_loss
            loss.backward()

            # keep track of the train loss for this step
            next_step = next(train_step)
            summary_writer.add_scalar(
                "revdict-train/cos",
                F.cosine_similarity(pred_embs, target_embs).mean().item(),
                next_step,
            )
            summary_writer.add_scalar("revdict-train/mse", mse_loss.item(), next_step)
            optimizer.step()
            pbar.update(target_embs.size(0))

        pbar.close()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss, sum_cosine, sum_rnk = 0.0, 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(valid_dataset),
                    disable=None,
                    leave=False,
                )
                pred_embs_list, target_embs_list = [], []
                for ids, word, gloss, electra, bertseg, bertmsa in valid_dataloader:
                    word_tokens = tokenizer(word, padding=True, return_tensors='pt').to(args.device)
                    gloss_tokens = tokenizer(gloss, padding=True, return_tensors='pt').to(args.device)
                    
                    # word_tokens = tokenizer(word, padding=True, return_tensors='pt').to(args.device)
                    # gloss_tokens = tokenizer(gloss, max_length=512, padding=True, truncation=True, return_tensors='pt').to(args.device)

                    if args.target_arch == "electra":
                        target_embs = torch.stack(electra, dim=1).to(args.device)
                    elif args.target_arch == "bertseg":
                        target_embs = torch.stack(bertseg, dim=1).to(args.device)
                    elif args.target_arch == "bertmsa":
                        target_embs = torch.stack(bertmsa, dim=1).to(args.device)

                    target_embs = target_embs.float()
                    ce_loss, pred_embs = model(
                        gloss_tokens["input_ids"], 
                        gloss_tokens["attention_mask"],
                        word_tokens["input_ids"],
                    )

                    mse_loss = loss_fn(pred_embs, target_embs)
                    loss = args.ce_loss_weight * ce_loss + mse_loss

                    sum_dev_loss += loss.item()
                    sum_cosine += F.cosine_similarity(pred_embs, target_embs).sum().item()
                    pred_embs_list.append(pred_embs.cpu())
                    target_embs_list.append(target_embs.cpu())
                    # sum_rnk += rank_cosine(pred_embs, target_embs)
                    pbar.update(target_embs.size(0))
                
                sum_rnk = rank_cosine(torch.cat(pred_embs_list, dim=0), torch.cat(target_embs_list, dim=0))
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch} cos: "+str(sum_cosine / len(valid_dataset))+" mse: "+str( sum_dev_loss / len(valid_dataset) )+" rnk: "+str(sum_rnk/ len(valid_dataset))+ " sum_rnk: "+str(sum_rnk)+" len of dev: "+str(len(valid_dataset)) +"\n",
                    total=len(valid_dataset),
                    disable=None,
                    leave=False,
                )

                if sum_cosine >= best_cosine:
                    best_cosine = sum_cosine
                    print(f"Saving Best Checkpoint at Epoch {epoch}")
                    model.save(args.save_dir / "model_best.pt")

                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(valid_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss / len(valid_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/rnk", sum_rnk / len(valid_dataset), epoch
                )
                pbar.close()
                model.train()

        model.save(args.save_dir / "modelepoch.pt")
            
    # 5. save result
    model.save(args.save_dir / "model.pt")






#|%%--%%| <kCSTM0MjE8|aSmKrELSlX>
class Args(object):
  def __init__(self, **kwargs):
    self.__dict__ = kwargs

# python revdict.py --do_train --train_file ../../../dev.json --dev_file ../../../dev.json  --model_name "aubmindlab/bert-base-arabertv02"

args_train = Args(
    # train_file = pathlib.Path("train.json"),
    train_file = pathlib.Path("dev.json"),
    # dev_file = pathlib.Path("dev.json"),
    save_dir = pathlib.Path("models/"),
    summary_logdir =pathlib.Path("logs/"),
    model_name="aubmindlab/bert-base-arabertv02",
    # device="cuda",
    device="cpu",
    target_arch="electra", # choices=("sgns", "electra", "bertseg", "bertmsa"),

    batch_size=64,
    resume_train=None,#"/content/drive/MyDrive/data_for_KKSA_NLP_CH/models/modelepoch.pt",
    resume_file=None,
    from_pretrained=False,
    max_len=256, # choices=(300, 256, 768),
    num_epochs=10
)


#|%%--%%| <aSmKrELSlX|gn451sUfOx>

train(args_train)

#|%%--%%| <gn451sUfOx|5kLNgpkrsN>

import pandas as pd
x =  pd.read_json("dev.json")

#|%%--%%| <5kLNgpkrsN|F1xrXeXHCW>

print(len(x.loc[0,"bertmsa"]))


# |%%--%%| <F1xrXeXHCW|d9qVMQEpnG>



