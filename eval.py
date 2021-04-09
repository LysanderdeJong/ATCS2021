# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import numpy as np
import logging
import sklearn
import argparse

from collections import Counter
from torchtext.vocab import Vocab

from pl_model import SentenceEmbeddings

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = './SentEval'
# path to the NLP datasets 
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')
# path to store GloVe vectors 
PATH_TO_VEC = os.path.join(PATH_TO_SENTEVAL, 'pretrained')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def create_dataset(senteces):
    counter = Counter()
    for sentece in senteces:
        counter.update(sentece)
    TEXT = Vocab(counter, specials=('<unk>', '<BOS>', '<EOS>', '<pad>'), vectors='glove.840B.300d')
    logging.info('Found {0} words with word vectors, out of {1} words'.format(len(TEXT.vectors), len(TEXT)))
    return TEXT
    
def create_model(params):
    model = SentenceEmbeddings(params.TEXT.vectors, **params.model_params)
    if params.model_params['checkpoint_path']:
        state_dict = torch.load(params.model_params['checkpoint_path'], map_location=model.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        del state_dict['embedding.weight']
        model.load_state_dict(state_dict, strict=False)
    if params.model_params['model'] != 'words':
        assert state_dict
    model.cuda()
    model.eval()
    return model


def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what your model needs into the params dictionary
    """
    params.TEXT = create_dataset(samples)
    params.model = create_model(params)
    params.w2i = params.TEXT.stoi
    params.process = lambda x: torch.tensor([params.w2i['<BOS>']] + [params.w2i[token] for token in x] + [params.w2i['<EOS>']])
    return

def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    tensor_data = list(map(params.process, batch))
    tensor_data = torch.nn.utils.rnn.pad_sequence(tensor_data, batch_first=True, padding_value=3)
    tensor_data = tensor_data.to(params.model.device)
    with torch.no_grad():
        sentence_embeddings = params.model.feature_forward(tensor_data)
    return sentence_embeddings.cpu().numpy()


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                        'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = SentenceEmbeddings.add_model_specific_args(parser)
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Load model paramaters from this file.')
    args = parser.parse_args()

    params_senteval['model_params'] = vars(args)
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']

    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    torch.save(results, f'results_{args.model}.pt')