import pandas as pd
import torch
import spacy
from torchtext import vocab as voc

df_caption = pd.read_csv("./dataset/csv/dataset_cleaned.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_caption = df_caption[:10000]

en_nlp = spacy.load("en_core_web_md")


def tokenize_comment(example):
    en_tokens = [token.text for token in en_nlp.tokenizer(example)][:100]
    en_tokens = [token.lower() for token in en_tokens]
    en_tokens = ["<sos>"] + en_tokens + ["<eos>"]
    return en_tokens


df_caption["text"] = df_caption["text"].apply(tokenize_comment)


min_freq = 1

unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

vocab = voc.build_vocab_from_iterator(
    df_caption["text"],
    min_freq=min_freq,
    specials=special_tokens,
)

unk_index = vocab[unk_token]
pad_index = vocab[pad_token]

vocab.set_default_index(unk_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(len(vocab))


def numericalize_str(example):
    vocab_ids = vocab.lookup_indices(example)
    return vocab_ids


df_caption["text"] = df_caption["text"].apply(numericalize_str)

no_worker = 64
