"""
finetune both models jointly


python joint/train.py     --train ../../data/v6/corpus.wordbiased.tag.train     --test ../../data/v6/corpus.wordbiased.tag.test.categories     --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 1     --pretrain_epochs 4     --learning_rate 0.0003 --epochs 2 --hidden_size 10 --train_batch_size 4 --test_batch_size 4     --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax --sequence_softmax     --working_dir TEST     --debug_skip
"""

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
import pickle
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import Counter
import math
import functools
import copy

from pytorch_pretrained_bert.modeling import BertEmbeddings
from pytorch_pretrained_bert.optimization import BertAdam


import sys; sys.path.append(".")
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import model as joint_model
import utils as joint_utils



working_dir = ARGS.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

with open(working_dir + '/command.sh', 'w') as f:
    f.write('python' + ' '.join(sys.argv) + '\n')

writer = SummaryWriter(ARGS.working_dir)



# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(
    ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab

tok2id['<del>'] = len(tok2id)
print("Vocab size: {}".format(len(tok2id)))

if ARGS.bert_encoder:
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
else:
    TRAIN_BATCH_SIZE = ARGS.train_batch_size
    TEST_BATCH_SIZE = ARGS.test_batch_size // ARGS.beam_width


if ARGS.pretrain_data:
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data,
        tok2id, TRAIN_BATCH_SIZE, ARGS.working_dir + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, TRAIN_BATCH_SIZE, ARGS.working_dir + '/train_data.pkl',
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, TEST_BATCH_SIZE, ARGS.working_dir + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)




# # # # # # # # ## # # # ## # # TAGGING MODEL # # # # # # # # ## # # # ## # #
# build model
if ARGS.extra_features_top:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tag_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')
if CUDA:
    tag_model = tag_model.cuda()

# train or load model
tagging_loss_fn = tagging_utils.build_loss_fn(debias_weight=1.0)


# # # # # # # # ## # # # ## # # DEBIAS MODEL # # # # # # # # ## # # # ## # #
# bulid model
if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)

if ARGS.freeze_bert and ARGS.bert_encoder:
    for p in debias_model.encoder.parameters():
        p.requires_grad = False
    for p in debias_model.embeddings.parameters():
        p.requires_grad = False

if ARGS.tagger_encoder:
    if ARGS.copy_bert_encoder:
        debias_model.encoder = copy.deepcopy(tag_model.bert)
        debias_model.embeddings = copy.deepcopy(tag_model.bert.embeddings.word_embeddings)
    else:
        debias_model.encoder = tag_model.bert
        debias_model.embeddings = tag_model.bert.embeddings.word_embeddings
if CUDA:
    debias_model = debias_model.cuda()

# train or load model
debias_loss_fn, cross_entropy_loss = seq2seq_utils.build_loss_fn(vocab_size=len(tok2id))


num_train_steps = (num_train_examples * 40)



joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tag_model)

if CUDA:
    joint_model = joint_model.cuda()

if ARGS.checkpoint is not None and os.path.exists(ARGS.checkpoint):
    print('LOADING FROM ' + ARGS.checkpoint)
    # TODO(rpryzant): is there a way to do this more elegantly? 
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    if CUDA:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint))
        joint_model = joint_model.cuda()
    else:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint, map_location='cpu'))
    print('...DONE')

model_parameters = filter(lambda p: p.requires_grad, joint_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)

debias_model= joint_model.debias_model

if ARGS.debias_checkpoint is not None and os.path.exists(ARGS.debias_checkpoint):
    print('LOADING DEBIASER FROM ' + ARGS.debias_checkpoint)
    debias_model.load_state_dict(torch.load(ARGS.debias_checkpoint))
    print('...DONE')



model=debias_model
optimizer = seq2seq_utils.build_optimizer(model, num_train_steps)
for epoch in range(ARGS.epochs):
    print('EPOCH ', epoch)
    print('TRAIN...')
    model.train()
    losses = seq2seq_utils.train_for_epoch(model, train_dataloader, tok2id, optimizer, debias_loss_fn, coverage=ARGS.coverage)
    writer.add_scalar('train/loss', np.mean(losses), epoch+1)

    print('SAVING...')
    model.save(ARGS.working_dir + '/model_%d.ckpt' % (epoch+1))

    print('EVAL...')
    model.eval()
    hits, preds, golds, srcs = seq2seq_utils.run_eval(
        model, eval_dataloader, tok2id, ARGS.working_dir + '/results_%d.txt' % epoch,
        ARGS.max_seq_len, ARGS.beam_width)
    # writer.add_scalar('eval/partial_bleu', utils.get_partial_bleu(preds, golds, srcs), epoch+1)
    writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), epoch+1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)




