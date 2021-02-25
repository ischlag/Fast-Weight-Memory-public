import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='PLASTICLSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU, PLASTICLSTM, MYLSTM, FASTPLASTICLSTM, SIMPLEPLASTICLSTM)')
parser.add_argument('--alphatype', type=str, default='full',
        help="type of alpha matrix: (full, perneuron, single)")
parser.add_argument('--modultype', type=str, default='none',
        help="type of modulation: (none, modplasth2mod, modplastc2mod)")
parser.add_argument('--modulout', type=str, default='single',
        help="modulatory output (single or fanout)")
parser.add_argument('--cliptype', type=str, default='clip',
                    help="clip type (decay, clip, aditya)")
parser.add_argument('--hebboutput', type=str, default='i2c',
                    help='output used for hebbian computations (i2c, h2co, cell, hidden)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--ssize', type=int, default=32, help='FWM s-size')
parser.add_argument('--rsize', type=int, default=32, help='FWM r-size')
parser.add_argument('--tsize', type=int, default=32, help='FWM t-size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--clipval', type=float, default=2.0,
                    help='value of the hebbian trace clipping')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--agdiv', type=float, default=1150.0,
                    help='divider of the gradient of alpha')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--proplstm', type=float, default=0.5,
                    help='for split-lstms: proportion of LSTM cells in the recurrent layer')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--asgdtime', type=int, default=-1,
                    help='number of iterations before switch to ASGD (if positive)')
parser.add_argument('--nonmono', type=int, default=5,
                    help='range of non monotonicity before switch to ASGD (if asgdtime is negative)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--numgpu', type=int, default=0,
                    help='which GPU to use? (no effect if GPU not used at all)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--embed_init', action='store_true',
                    help='')
parser.add_argument('--embed_model', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--prefix', type=str,  default='none',
                    help='path of model to resume')
parser.add_argument('--asgd_lr', type=float, default=0.5,
                    help='lr for asgd')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='logs dir')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  if  not args.cuda :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  else:
    torch.cuda.manual_seed(args.seed)
else:
  print("NOTE: no CUDA device detected.")

import platform
print("PyTorch version:", torch.__version__,
      "Numpy version:", np.version.version,
      "Python version:", platform.python_version(),
      "GPU used (if any):", args.numgpu)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, test_batch_size, args)
val_data = batchify(corpus.valid, test_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


#train_data = train_data[:5000,:]   # For debugging

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)

# Configuration parameters of the plastic LSTM. See mylstm.py for details.
myparams={}
myparams['clipval'] = args.clipval
myparams['cliptype'] = args.cliptype
myparams['modultype'] = args.modultype
myparams['modulout'] = args.modulout
myparams['hebboutput'] = args.hebboutput
myparams['alphatype'] = args.alphatype
myparams['s_size'] = args.ssize
myparams['r_size'] = args.rsize
myparams['t_size'] = args.tsize

suffix = ('_SqUsq_' + args.model
          + '_' + myparams['cliptype']
          + '_cv' + str(myparams['clipval'])
          + '_' + myparams['modultype']
          + '_' + myparams['modulout']
          + '_' + myparams['hebboutput']
          + '_' + myparams['alphatype']
          + '_asgdtime' + str(args.asgdtime)
          + '_agdiv' + str(int(args.agdiv))
          + '_lr' + str(args.lr)
          + '_' + str(args.nlayers)
          + 'l_' + str(args.nhid)
          + 'h_' + str(args.proplstm)
          + 'lstm_rngseed' + str(args.seed))
print("Suffix:", suffix)
MODELFILENAME = args.log_dir + '/model_' + args.prefix + suffix + '.dat'
RESULTSFILENAME = args.log_dir + '/results_'+ args.prefix + suffix + '.txt'
FILENAMESTOSAVE = [MODELFILENAME, RESULTSFILENAME]  # We will append to this list the additional files at each learning rate reduction, if any

print("Plasticity and neuromodulation parameters:", myparams)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.proplstm, args.nlayers, args.dropout,
                       args.dropouth, args.dropouti, args.dropoute,
                       args.wdrop, args.tied, myparams)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    # if args.wdrop:
    #     from weight_drop import WeightDrop
    #     for rnn in model.rnns:
    #         if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
    #         elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

if args.embed_init:
    print('Using pretrained embeds:', args.embed_model)
    with open(args.embed_model, 'rb') as f:
        embed_model = torch.load(f)
    if type(embed_model) == list:
        # criterion = embed_model[1]
        embed_model = embed_model[0]

    model.decoder = embed_model.decoder
    model.encoder = embed_model.encoder

    del embed_model

###
params = list(model.parameters()) + list(criterion.parameters())
if args.cuda:
    model = model.cuda(args.numgpu)
    criterion = criterion.cuda(args.numgpu)
    params = list(model.parameters()) + list(criterion.parameters())
###
total_params = sum(x.numel() for x in params if x.numel())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10, temperature=1.0):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        # model.reset()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)

            output = output / temperature

            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
temperature = 1.08

# Load the best saved model.
model_load(MODELFILENAME)
print("model loaded from", MODELFILENAME)

best_temp = 0.8
best_ppl = 9999

for temperature in np.arange(0.8, 1.2, 0.01):
    # Run on validation data.
    print("temperature: ", temperature)
    val_loss = evaluate(val_data, test_batch_size, temperature)
    print('valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
        val_loss, math.exp(val_loss), val_loss / math.log(2)))

    ppl = math.exp(val_loss)
    if ppl < best_ppl:
        best_ppl = ppl
        best_temp = temperature

    if ppl > best_ppl:
        break

# Run on test data.
temperature = best_temp
print("best temperature: ", temperature)
test_loss = evaluate(test_data, test_batch_size, temperature)
print('test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))

# Run on validation data.
val_loss = evaluate(val_data, test_batch_size, temperature)
print('valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
    val_loss, math.exp(val_loss), val_loss / math.log(2)))

# Run on train data.
train_loss = evaluate(train_data, test_batch_size, temperature)
print('train loss {:5.2f} | train ppl {:8.2f} | train bpc {:8.3f}'.format(
    train_loss, math.exp(train_loss), train_loss / math.log(2)))