import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy.datasets import SNLI
from torchtext.legacy.data import Field, LabelField, BucketIterator

def create_dataset():
    TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='<BOS>', eos_token='<EOS>', lower=True, batch_first=True)
    LABEL = LabelField(dtype=torch.long)
    train, val, test = SNLI.splits(TEXT, LABEL)
    TEXT.build_vocab(train, vectors='glove.840B.300d')
    LABEL.build_vocab(train)
    return train, val, test, TEXT, LABEL