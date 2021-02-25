import math
import torch
import torch.nn as nn
import torch.jit as jit

from utils.lib import grow_batch, traverse, check
from torch.nn.functional import relu as ramp
from typing import Tuple, List
from torch import Tensor

MIN_N_GPUS = 4

def load_default_params(p):
    # size of the vocabulary
    p["vocab_size"] = 180
    # size of the token embedding vector
    p["embedding_size"] = 256
    # size of the lstm hidden state and lstm output
    p["hidden_size"] = 256
    # size of the source node dimension of the TPR
    p["s_size"] = 32
    # size of the relation dimension of the TPR
    p["r_size"] = 32
    # size of the target node dimension of the TPR
    p["t_size"] = 32
    # number of sequential reads
    p["n_reads"] = 3
    # residual
    p["residual"] = True

def get_string_description(p):
  txt = "FWM_emb{}_h{}_s{}_r{}_t{}_reads{}{}"
  return txt.format(
    p["embedding_size"],
    p["hidden_size"],
    p["s_size"],
    p["r_size"],
    p["t_size"],
    p["n_reads"],
    "_Res" if p["residual"] else "_nonRes",
  )


class Model(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.p = p = params
    self.QM = p.QM
    self.PAD = p.PAD
    self.embedding = nn.Embedding(p.vocab_size, p.embedding_size)

    cells = []
    cells.append(TPRRNNCell(p.embedding_size,
                            p.hidden_size,
                            p.s_size,
                            p.r_size,
                            p.t_size,
                            p))
    cells = torch.nn.ModuleList(cells)
    self.rnn = RNN(cells=cells, params=p)

    self.linear = nn.Linear(p.hidden_size, p.vocab_size, bias=False)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_normal_(self.linear.weight, gain=1.0)

  def forward(self, x, seq_len, hidden_state=None):
    bs, sl = x.shape[0], x.shape[1]
    check(x, [bs, sl])
    check(seq_len, [bs])

    if self.p.ra_mode:
      ra_pos = x == self.QM
    else:
      ra_pos = x != self.PAD

    pad_step = x == self.PAD

    embedding = self.embedding(x)
    check(embedding, [bs, sl, self.p.embedding_size])

    out, hidden_state = self.rnn(embedding, seq_len, ra_pos, pad_step, hidden_state)
    check(out, [bs, sl, self.p.hidden_size])

    logits = self.linear(out)
    check(logits, [bs, sl, self.p.vocab_size])

    regularizer = self.rnn.cells[0].F_norm

    # returns should have batch_size as the first dimension
    return logits, regularizer, hidden_state


class RNN(nn.Module):
  def __init__(self, cells, params):
    super().__init__()
    self.p = params
    self.timedim = 1
    self.cells = cells
    self.n_layers = len(cells)

  def forward(self, x, seq_len, ra_pos, pad_step, hidden_state):
    # x: [batch_size, seq_length, embedding_size]
    # seq_len: [batch_size]
    # ra_pos: [batch_size, seq_length]
    # pad_step: [batch_size, seq_length]
    batch_size = x.shape[0]

    # set initial state if hidden_state is None
    if hidden_state:
      states = hidden_state
    else:
      states = []
      for cell in self.cells:
        states.append(cell.get_initial_state(batch_size, x.device))

    x_list = x.unbind(dim=self.timedim)
    ra_pos_list = ra_pos.unbind(dim=self.timedim)
    pad_step_list = pad_step.unbind(dim=self.timedim)
    outputs = []

    # loop over seq_length
    t = 0
    for curr_x in x_list:
      curr_ra_pos = ra_pos_list[t]
      curr_pad_step = pad_step_list[t].float() # 1.0 if pad, 0.0 if non-pad
      curr_bs = torch.sum(seq_len > t)
      compute_bs = MIN_N_GPUS if curr_bs < MIN_N_GPUS else curr_bs  # need at least 1 sample per gpu
      # curr_bs: []

      # curr_x: [batch_size, embedding_size]
      new_states = []
      layer = 0
      # loop over layers
      for cell in self.cells:
        # shorten batch according to sequence length
        reduce_batch = lambda t: t[:compute_bs]
        state = traverse(states[layer], reduce_batch)

        # state: (lstm_state, fw_state)
        curr_x, new_state = cell(curr_x[:compute_bs],
                                 state,
                                 curr_ra_pos[:compute_bs])
        # curr_x: [batch_size, hidden_size]
        # new_state: (lstm_state, fw_state)

        # update the processed states and keep the previous ones
        if curr_bs != batch_size:
          # if this is a smaller bs due to the end of epoch, keep the old states
          new_lstm_0 = new_state[0][0]
          new_lstm_1 = new_state[0][1]
          new_fw = new_state[1]

          prev_lstm_0 = states[layer][0][0]
          prev_lstm_1 = states[layer][0][1]
          prev_fw = states[layer][1]

          new_state = (
            (torch.cat([new_lstm_0[:curr_bs], prev_lstm_0[curr_bs:]], dim=0), 
             torch.cat([new_lstm_1[:curr_bs], prev_lstm_1[curr_bs:]], dim=0)),
            torch.cat([new_fw[:curr_bs], prev_fw[curr_bs:]], dim=0)
          )

        new_states += [new_state]
        layer = layer + 1

      curr_x = grow_batch(curr_x, batch_size)
      outputs += [curr_x]
      # carry states to the next time step if there is a pad input
      lstm_pad_step = curr_pad_step.unsqueeze(1)
      fwm_pad_step = curr_pad_step.unsqueeze(1).unsqueeze(1).unsqueeze(1)
      for l in range(len(states)):  # only update state of non-pad input
        states[l] = ((states[l][0][0] * lstm_pad_step + new_states[l][0][0] * (1-lstm_pad_step),  # lstm state 0
                      states[l][0][1] * lstm_pad_step + new_states[l][0][1] * (1-lstm_pad_step)),  # lstm state 1
                     states[l][1] * fwm_pad_step + new_states[l][1] * (1-fwm_pad_step)) # fwm state
      t = t + 1
    # states: list of ([batch_size, memory_size], lstm_state, fw_state)

    outputs = torch.stack(outputs, dim=self.timedim)
    # outputs: [batch_size, seq_length, t_size]

    return outputs, states


class TPRRNNCell(nn.Module):
  def __init__(self, input_size, hidden_size,
               s_size, r_size, t_size, p):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.s_size = s_size
    self.r_size = r_size
    self.t_size = t_size
    self.b_size = 1
    self.p = p
    self.n_reads = p.n_reads

    self.lstm = LSTMCell(input_size=input_size, hidden_size=hidden_size)

    # write
    write_size = s_size + r_size + t_size + self.b_size
    self.W_write = nn.Linear(hidden_size, write_size)

    # read
    self.W_read = nn.Linear(hidden_size, s_size + r_size * self.n_reads)
    self.ln_read = nn.LayerNorm(t_size, elementwise_affine=False)
    self.linear = nn.Linear(t_size, hidden_size)

    # inital fast weight state
    self.get_initial_fw_state = lambda: torch.zeros(self.s_size, self.r_size, self.t_size)
    self.reset_parameters()

  def get_initial_state(self, batch_size: int, device):
    return (
      self.lstm.get_initial_state(batch_size, device),
      torch.stack([self.get_initial_fw_state().to(device)] * batch_size, dim=0),
    )

  def reset_parameters(self):
    a = math.sqrt(6. / (self.hidden_size + self.s_size))
    nn.init.uniform_(self.W_write.weight, -a, a)
    nn.init.uniform_(self.W_read.weight, -a, a)

    # set the forget gate bias of the lstm to 1
    self.lstm.reset_parameters()

  def forward(self, x, state, ra_pos):
    # x: [batch_size, embedding_size]
    # state: (lstm_state, [batch_size, s_size, r_size, t_size])
    # ra_pos: [batch_size]
    lstm_state, F = state

    z, new_lstm_state = self.lstm(x, lstm_state)
    # z: [batch_size, hidden_size]
    # lstm_new_state: ([batch_size, hidden_size],
    #                  [batch_size, hidden_size])

    # --- write to memory ----
    write_vars = self.W_write(z)
    # write_vars: [batch_size, s_size + r_size + t_size + b_size]

    write_sizes = [self.s_size, self.r_size, self.t_size, self.b_size]
    s, r, t, b = torch.split(write_vars, write_sizes, dim=1)

    s = torch.tanh(s)
    r = torch.tanh(r)
    t = torch.tanh(t)
    b = torch.sigmoid(b + 1)  # bias to writing

    sr = torch.einsum("bs,br->bsr", s, r)
    # sr: [batch_size, s_size, r_size]

    v = torch.einsum("bsr,bsrv->bv", sr, F)
    # v: [batch_size, t_size]

    new_v = b.view(-1, 1) * (t - v) / self.p.t_size

    F = F + torch.einsum("bsr,bv->bsrv", sr, new_v)

    self.F_norm = F.view(F.shape[0], -1).norm(dim=-1)
    fnorm = torch.relu(self.F_norm - 1) + 1
    F = F / fnorm.view(-1, 1, 1, 1)

    # --- read from memory ----
    output = torch.zeros((z.shape[0], self.hidden_size)).to(z.device)
    if ra_pos.any():
      n_read_args = torch.split(self.W_read(z), 
                                [self.s_size] + [self.r_size] * self.n_reads,
                                dim=1)
      q, n_read_args = n_read_args[0], n_read_args[1:]

      for r in n_read_args:
        r = torch.tanh(r)
        q = torch.einsum("bsrv,bs,br->bv", F, q, r)
        q = self.ln_read(q)
        # q: [batch_size, t_size]
      
      output = self.linear(q)
      # output: [batch_size, hidden_size]
    
    if self.p.residual:
      output = z + output

    new_state = (new_lstm_state, F)
    return output, new_state


class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.W_ih = nn.Parameter(torch.rand(4*hidden_size, input_size))
    self.W_hh = nn.Parameter(torch.rand(4*hidden_size, hidden_size))
    self.bias = nn.Parameter(torch.rand(4*hidden_size))
    self.register_parameter("W_ih", self.W_ih)
    self.register_parameter("W_hh", self.W_hh)
    self.register_parameter("bias", self.bias)
    self.reset_parameters()

  def get_initial_state(self, batch_size: int, device) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(batch_size, self.hidden_size).to(device),
      torch.zeros(batch_size, self.hidden_size).to(device)
    )

  def reset_parameters(self):
    std = 1.0 / math.sqrt(self.hidden_size)
    self.W_ih.data.uniform_(-std, std)
    self.W_hh.data.uniform_(-std, std)
    self.bias.data.fill_(0)

  def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]):
    # input: [batch_size, input_size]
    # state: ([batch_size, hidden_size], [batch_size, hidden_size])
    h, c = state

    # mm is faster than matmul
    gates = torch.mm(x, self.W_ih.t()) + torch.mm(h, self.W_hh.t()) + self.bias
    # gates: [batch_size, 4*hidden_size]

    i, f, z, o = torch.chunk(gates, chunks=4, dim=1)
    # i,f,z,o: [batch_size, hidden_size]

    i = torch.sigmoid(i)
    f = torch.sigmoid(f + 1)
    z = torch.tanh(z)
    o = torch.sigmoid(o)

    new_c = (f * c) + (i * z)
    new_h = o * torch.tanh(new_c)

    return new_h, (new_h, new_c)
