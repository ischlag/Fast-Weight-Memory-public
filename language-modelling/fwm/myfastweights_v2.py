import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import LayerNorm
import torch.nn.functional as F
import numpy as np

from weight_drop import WeightDrop

import pdb


class FWMRNN(nn.Module):
  def __init__(self, isize, hsize, withFWM, params, wdrop=0.5):
    super().__init__()
    s_size = params["s_size"]
    r_size = params["r_size"]
    t_size = params["t_size"]
    self.rnn = nn.LSTM(isize, hsize, 1, dropout=0)

    if withFWM:
      self.fwm = FWM(hsize, s_size, r_size, t_size)
      self.linear = nn.Linear(t_size, hsize)

    self.isize = isize
    self.hsize = hsize
    self.hasFWM = withFWM

    self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=wdrop)

  def reset(self):
    pass

  def forward(self, inputs, hidden):
    lstm_hidden, F = hidden

    x, lstm_hidden = self.rnn(inputs, lstm_hidden)
    outputs = []
    if self.hasFWM:
      for t, x_t in enumerate(x):
        F = self.fwm.write(x_t, F)
        o_t = self.fwm(x_t, F)
        outputs.append(o_t)
      s = torch.stack(outputs, dim=0)
      output = x + self.linear(s)
    else:
      output = x

    hidden = (lstm_hidden, F)
    return output, hidden


class FWM(nn.Module):
  def __init__(self, hidden_size, s_size, r_size, t_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.s_size = s_size
    self.r_size = r_size
    self.t_size = t_size
    self.b_size = 1

    write_size = s_size + r_size + t_size + self.b_size
    self.W_write = nn.Linear(hidden_size, write_size)

    # read
    self.W_read = nn.Linear(hidden_size, s_size + r_size * 3)
    self.ln_read = LayerNorm(t_size, elementwise_affine=False)

    self.reset_parameters()

  def reset_parameters(self):
    a = math.sqrt(6. / (self.hidden_size + self.s_size))
    nn.init.uniform_(self.W_write.weight, -a, a)
    nn.init.uniform_(self.W_read.weight, -a, a)

  def write(self, z, F):
    # z: [batch_size, hidden_size]
    # F: [batch_size, s_size, r_size, t_size]

    write_vars = self.W_write(z)
    # write_vars: [batch_size, n_writes * (s_size + r_size + t_size + b_size)]

    write_sizes = [self.s_size, self.r_size, self.t_size, self.b_size]
    write_list = torch.split(write_vars, sum(write_sizes), dim=1)
    # write_list: list of [batch_size, s_size + r_size + t_size + b_size]

    # multiple writes at once
    # scale = 1./self.t_size
    scale = 1./(3*self.t_size)
    for write_idx, write_el in enumerate(write_list):
      s, r, t, b = torch.split(write_el, write_sizes, dim=1)
      s = torch.tanh(s)
      r = torch.tanh(r)
      t = torch.tanh(t)
      # *: [batch_size, *_size]

      b = torch.sigmoid(b + 1)
      # b: [batch_size, 1]

      # sr = torch.einsum("bs,br->bsr", s, r)
      sr = s.unsqueeze(2) * r.unsqueeze(1)
      sr = sr.view(sr.shape[0], -1)
      # sr: [batch_size, s_size, r_size]
      # v = torch.einsum("bsr,bsrv->bv", sr, F)
      v = torch.matmul(sr.unsqueeze(1), F).squeeze(1)

      # v: [batch_size, t_size]
      new_v = b.view(-1, 1) * (t - v)

      # new_v: [batch_size, t_size]
      # F = F + torch.einsum("bsr,bv->bsrv", sr, new_v * scale)
      delta = sr.unsqueeze(2) * new_v.unsqueeze(1)
      F = F + delta

    # scale F down if norm is > 1
    F_norm = F.view(F.shape[0], -1).norm(dim=-1)
    F_norm = torch.relu(F_norm - 1) + 1
    F = F / F_norm.unsqueeze(1).unsqueeze(1)

    return F

  def forward(self, z, F):
    read_vars = self.W_read(z)
    read_vars = torch.tanh(read_vars)
    # read_vars: [batch_size, s_size + r_size + r_size]
    n_read_args = torch.split(
                  read_vars,
                  [self.s_size] + [self.r_size] * 3,
                  dim=1)
    q, n_read_args = n_read_args[0], n_read_args[1:]

    for r in n_read_args:
      # q = torch.einsum("bsrv,bs,br->bv", F, q, r)
      # q = self.ln_read(q)
      nr = q.unsqueeze(2) * r.unsqueeze(1)
      nr = nr.view(nr.shape[0], -1)
      q = torch.matmul(nr.unsqueeze(1), F).squeeze(1)
      q = self.ln_read(q)

    return q