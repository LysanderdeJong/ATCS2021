import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import math

from pytorch_lightning.metrics.functional import accuracy
from modules import Words, LSTM


class SentenceEmbeddings(pl.LightningModule):
    def __init__(self, VECTORS, input_size=300, hidden_size=2048, num_classes=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        del self._hparams_initial['VECTORS']
        del self._hparams['VECTORS']

        self.embedding = nn.Embedding.from_pretrained(VECTORS)
        
        if self.hparams.model == 'words':
            hidden_size = input_size
            self.hparams.hidden_size = self.hparams.input_size
            self.sentence_embedding = Words()
        else:
            self.sentence_embedding = LSTM(self.hparams.model, input_size=self.hparams.input_size,
                                           hidden_size=self.hparams.hidden_size,
                                           dropout=self.hparams.dropout,
                                           bidirectional=self.hparams.bidirectional)
        assert self.sentence_embedding
        if 'bilstm' in self.hparams.model:
            hidden_size *= 2
        
        self.mlp = self.mlp = nn.Sequential(
                      nn.Linear(hidden_size*4, 512),
                      nn.ReLU(),
                      nn.Linear(512, num_classes),
                    ) if num_classes > 0 else nn.Identity()
        
        self.criterion = nn.CrossEntropyLoss()
        
    def feature_forward(self, x):
        x = self.embedding(x)
        x = self.sentence_embedding(x)
        return x
    
    def forward(self, x):
        premise, hypothesis = x
        premise = self.feature_forward(premise)
        hypothesis = self.feature_forward(hypothesis)
        x = self.mlp(torch.cat([premise, hypothesis, torch.abs(premise - hypothesis), premise*hypothesis], dim=-1))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, batch, batch_indx):
        loss, pred = self.step(batch, batch_indx)
        acc = accuracy(F.softmax(pred, dim=-1), batch.label)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("acc", acc, on_step=True, on_epoch=False, logger=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_indx):
        loss, pred = self.step(batch, batch_indx)
        acc = accuracy(F.softmax(pred, dim=-1), batch.label)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_indx):
        loss, pred = self.step(batch, batch_indx)
        acc = accuracy(F.softmax(pred, dim=-1), batch.label)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return loss

    def step(self, batch, batch_indx=None):
        pred = self.forward((batch.premise, batch.hypothesis))
        loss = self.criterion(pred, batch.label)
        return loss, pred
    
    def reset_mlp(self, embed_dim, num_classes, activation=nn.ReLU()):
        self.mlp = nn.Sequential(
                      nn.Linear(embed_dim, 512),
                      activation,
                      nn.Linear(512, num_classes),
                    ) if num_classes > 0 else nn.Identity()
        
    def reset_embedding(self, vectors):
        self.embedding = nn.Embedding.from_pretrained(vectors)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SentenceEmbeddingsModel")
        
        # model parameters
        parser.add_argument('--model', default='words', type=str,
                        choices=['words', 'lstm', 'bilstm', 'bilstm_max'],
                        help='Which model to train.')
        parser.add_argument('--input_size', default=300, type=int,
                            help='Input size of the lstm module, should be the size of the embeddings.')
        parser.add_argument('--hidden_size', default=2048, type=int,
                            help='Hidden size on the LSTM module.')
        parser.add_argument('--num_classes', default=3, type=int,
                            help='Number of classes in the mlp for classification.')
        parser.add_argument('--dropout', default=0.0, type=float,
                            help='Dropout on the LSTM module.')
        parser.add_argument('--bidirectional', action='store_true',
                            help='Dropout on the LSTM module.')

        # Optimizer hyperparameters
        parser.add_argument('--lr', default=1e-1, type=float,
                            help='Learning rate to use.')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='Weight decay.')
        
        return parent_parser