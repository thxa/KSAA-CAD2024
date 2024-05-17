import math
from collections import defaultdict
from itertools import count
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput





BOS = "<seq>"
EOS = "</seq>"
PAD = "<pad/>"
UNK = "<unk/>"

SUPPORTED_ARCHS = (["sgns"])

# A dataset is a container object for the actual data
class JSONDataset(Dataset):
    """Reads a CODWOE JSON dataset"""


    def __init__(self, file, vocab=None, freeze_vocab=False, maxlen=256):
        """
        Construct a torch.utils.data.Dataset compatible with torch data API and
        codwoe data.
        args: `file` the path to the dataset file
              `vocab` a dictionary mapping strings to indices
              `freeze_vocab` whether to update vocabulary, or just replace unknown items with OOV token
              `maxlen` the maximum number of tokens per gloss
        """
        if vocab is None:
            self.vocab = defaultdict(count().__next__)
        else:
            self.vocab = defaultdict(count(len(vocab)).__next__)
            self.vocab.update(vocab)
        pad, eos, bos, unk = (
            self.vocab[PAD],
            self.vocab[EOS],
            self.vocab[BOS],
            self.vocab[UNK],
        )

        
        if freeze_vocab:
            self.vocab = dict(vocab)
        with open(file, "r") as istr:
            self.items = json.load(istr)

        # preparse data
        self.tensors = torch.tensor([])
        for json_dict in self.items:
            # in definition modeling test datasets, gloss targets are absent
            if "gloss" in json_dict:
                json_dict["gloss_tensor"] = torch.tensor(
                    [bos]
                    + [
                        self.vocab[word]
                        if not freeze_vocab
                        else self.vocab.get(word, unk)
                        for word in json_dict["gloss"].split()
                    ]
                    + [eos]
                )
                if maxlen:
                    # sz = json_dict["gloss_tensor"].size()[0]
                    # if(sz < maxlen):
                        # padding_data = torch.tensor([pad]*(maxlen-sz))
                        # json_dict["gloss_tensor"] = torch.cat((json_dict["gloss_tensor"], padding_data), dim=0)

                    json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]


            # in reverse dictionary test datasets, vector targets are absent
            for arch in SUPPORTED_ARCHS:
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])


            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])
                self.tensors = torch.cat((self.tensors, json_dict["electra_tensor"].unsqueeze(0)), dim=0)

            elif "bertseg" in json_dict:
                json_dict["bertseg_tensor"] = torch.tensor(json_dict["bertseg"])
            elif "bertmsa" in json_dict:
                json_dict["bertmsa_tensor"] = torch.tensor(json_dict["bertmsa"])



        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = SUPPORTED_ARCHS[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]
        self.has_bertseg = "bertseg" in self.items[0]
        self.has_bertmsa = "bertmsa" in self.items[0]
        self.itos = sorted(self.vocab, key=lambda w: self.vocab[w])
    
    

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    # we're adding this method to simplify the code in our predictions of
    # glosses
    def decode(self, tensor):
        """Convert a sequence of indices (possibly batched) to tokens"""
        with torch.no_grad():
            if tensor.dim() == 2:
                # we have batched tensors of shape [Seq x Batch]
                decoded = []
                for tensor_ in tensor.t():
                    decoded.append(self.decode(tensor_))
                return decoded
            else:
                return " ".join(
                    [self.itos[i.item()] for i in tensor if i != self.vocab[PAD]]
                )

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)


# A sampler allows you to define how to select items from your Dataset. Torch
# provides a number of default Sampler classes
class TokenSampler(Sampler):
    """Produce batches with up to `batch_size` tokens in each batch"""

    def __init__(
        self, dataset, batch_size=200, size_fn=len, drop_last=False, shuffle=True
    ):
        """
        args: `dataset` a torch.utils.data.Dataset (iterable style)
              `batch_size` the maximum number of tokens in a batch
              `size_fn` a callable that yields the number of tokens in a dataset item
              `drop_last` if True and the data can't be divided in exactly the right number of batch, drop the last batch
              `shuffle` if True, shuffle between every iteration
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        longest_len = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                if selected:
                    yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = (
                sum(self.size_fn(self.dataset[i]) for i in range(len(self.dataset)))
                // self.batch_size
            )
        return self._len


# DataLoaders give access to an iterator over the dataset, using a sampling
# strategy as defined through a Sampler.
def get_dataloader(dataset, batch_size=200, shuffle=True):
    """produce dataloader.
    args: `dataset` a torch.utils.data.Dataset (iterable style)
          `batch_size` the maximum number of tokens in a batch
          `shuffle` if True, shuffle between every iteration
    """
    # some constants for the closures
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra
    has_bertseg = dataset.has_bertseg
    has_bertmsa = dataset.has_bertmsa
    PAD_idx = dataset.vocab[PAD]

    # the collate function has to convert a list of dataset items into a batch
    def do_collate(json_dicts):
        """collates example into a dict batch; produces ands pads tensors"""
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        print(batch)
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"], padding_value=PAD_idx, batch_first=False
            )
        if has_vecs:
            for arch in SUPPORTED_ARCHS:
                batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
        if has_electra:
            batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        if has_bertseg:
            batch["bertseg_tensor"] = torch.stack(batch["bertseg_tensor"])
        if has_bertmsa:
            batch["bertmsa_tensor"] = torch.stack(batch["bertmsa_tensor"])
        return dict(batch)

    if dataset.has_gloss:
        # we try to keep the amount of gloss tokens roughly constant across all
        # batches.
        def do_size_item(item):
            """retrieve tensor size, so as to batch items per elements"""
            return item["gloss_tensor"].numel()

        return DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=TokenSampler(
                dataset, batch_size=batch_size, size_fn=do_size_item, shuffle=shuffle
            ),
        )
    else:
        # there's no gloss, hence no gloss tokens, so we use a default batching
        # strategy.
        return DataLoader(
            dataset, collate_fn=do_collate, batch_size=batch_size, shuffle=shuffle
        )


class AraT5RevDict(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.resume_train:
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_file)
            raise NotImplementedError()
        else:
            if args.from_pretrained:
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/AraT5v2-base-1024")
            else:
                model_config = AutoConfig.from_pretrained("UBC-NLP/AraT5v2-base-1024")
                self.base_model = AutoModelForSeq2SeqLM.from_config(model_config)
        
        self.linear = nn.Linear(self.base_model.config.hidden_size, args.max_len)

    def forward(self, input_ids, attention_mask, labels):
        outputs:Seq2SeqLMOutput = self.base_model(input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )        

        pooled_emb = (outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1)

        embedding = self.linear(pooled_emb)
        return outputs.loss, embedding     

    def save(self, file):
        torch.save(self, file)
        print("\n--\nsave1\n--\n")

    @staticmethod
    def load(file):
        return torch.load(file)

class ARBERTRevDict(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.resume_train:
            self.base_model = AutoModel.from_pretrained(args.resume_file)
            raise NotImplementedError()
        else:
            if args.from_pretrained:
                self.base_model = AutoModel.from_pretrained(args.model_name)
            else:
                model_config = AutoConfig.from_pretrained(args.model_name)
                self.base_model = AutoModel.from_config(model_config)
        
        self.linear = nn.Linear(self.base_model.config.hidden_size, args.max_len)

    def forward(self, input_ids, token_type_ids , attention_mask):
        feats = self.base_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        embedding = self.linear(feats)
        return embedding     

    def save(self, file):
        self.base_model.save_pretrained(file,from_pt=True)
        print("\n--\nsave_pretrained\n--\n")
        # torch.save(self, file)

    @staticmethod
    def load(file):
        return AutoModel.from_pretrained(file)
    
class PositionalEncoding(nn.Module):
    """From PyTorch"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)



class RevdictModel(nn.Module):
    """A transformer architecture for Reverse Dictionary"""

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512
    ):
        super(RevdictModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[PAD]
        self.eos_idx = vocab[EOS]
        self.maxlen = maxlen
        # self.vocab=vocab
        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_model, d_model)
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.dropout(
            self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask.t())
        )
        summed_embs = transformer_output.masked_fill(
            src_key_padding_mask.unsqueeze(-1), 0
        ).sum(dim=0)
        return self.e_proj(F.relu(summed_embs))

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)
        print("\n--\nsave2\n--\n")
