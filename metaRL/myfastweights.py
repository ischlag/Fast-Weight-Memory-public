import torch
import math
from torch import nn
from torch.autograd import Variable
from torch.nn import LayerNorm
import torch.nn.functional as F
import numpy as np
import pdb


class FWMRNN(nn.Module):
  def __init__(self, isize, hsize, withFWM=True, s_size=32):
    super().__init__()
    s_size = s_size#params["s_size"]
    r_size = s_size#params["r_size"]
    t_size = s_size#params["t_size"]
    self.rnn = nn.LSTM(isize, hsize, 1, dropout=0)

    if withFWM:
      self.fwm = FWM(hsize, s_size, r_size, t_size)
      self.linear = nn.Linear(t_size, hsize)

    self.isize = isize
    self.hsize = hsize
    self.hasFWM = withFWM

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
    self.W_write = nn.Linear(hidden_size, 1 * write_size)

    # read
    self.W_read = nn.Linear(hidden_size, s_size + r_size * 3)
    self.ln_read = LayerNorm(t_size, elementwise_affine=False)

    self.reset_parameters()

  def reset_parameters(self):
    a = 1.0 * math.sqrt(6. / (self.hidden_size + self.s_size))
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
    scale = 1./self.t_size
    for write_idx, write_el in enumerate(write_list):
      s, r, t, b = torch.split(write_el, write_sizes, dim=1)
      s = torch.tanh(s)
      r = torch.tanh(r)
      t = torch.tanh(t)
      # *: [batch_size, *_size]

      b = torch.sigmoid(b + 1)
      # b: [batch_size, 1]

      sr = torch.einsum("bs,br->bsr", s, r)
      ## sr: [batch_size, s_size, r_size]

      v = torch.einsum("bsr,bsrv->bv", sr, F)
      ## v: [batch_size, t_size]

      new_v = b.view(-1, 1) * (t - v)
      # new_v: [batch_size, t_size]

      F = F + torch.einsum("bsr,bv->bsrv", sr, new_v * scale)

    # scale F down if norm is > 1
    F_norm = F.view(F.shape[0], -1).norm(dim=-1)
    F_norm = torch.relu(F_norm - 1) + 1
    F = F / F_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    
    return F

  def forward(self, z, F):
    read_vars = self.W_read(z)
    # read_vars: [batch_size, s_size + r_size*n_reads]
    n_read_args = torch.split(
                  read_vars,
                  [self.s_size] + [self.r_size] * 3,
                  dim=1)  # [n, r_a, r_b, r_c]
    q, n_read_args = n_read_args[0], n_read_args[1:]

    for i, r in enumerate(n_read_args):
      q = torch.einsum("bsrv,bs,br->bv", F, q, r)
      q = self.ln_read(q)

    return q
