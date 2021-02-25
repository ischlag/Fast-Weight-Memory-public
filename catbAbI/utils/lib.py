import logging
import importlib
import os
import shutil
import time
import pickle
import csv
import torch
import random
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

NAN_CHECK_ON = False

def check(tensor, shape):
    """ Checks the shape of the tensor for better code redability and bug prevention. """
    assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: tensor is not torch.Tensor!"
    tensor_shape = list(tensor.shape)
    assert len(shape) == len(tensor_shape), f"SHAPE GUARD: tensor shape {tensor_shape} not the same length as {shape}"
    
    for idx, (a, b) in enumerate(zip(tensor_shape, shape)):
        if b <= 0:
            continue  # ignore -1 sizes
        else:
            assert a == b, f"SHAPE GUARD: at pos {str(idx)}, tensor shape {tensor_shape} does not match {shape}"


def nan_checker(x, tag="None"):
    if NAN_CHECK_ON:
        assert torch.isnan(x).sum().item() == 0, f"NaN: {x.shape} {tag}"
        assert torch.isinf(x).sum().item() == 0, f"INF: {x.shape} {tag}"


def assert_entries_exist(map, keys):
  """ raises an attribute error if any on the keys does not exist. """
  for k in keys:
    if k not in map.__dict__.keys():
      raise AttributeError("Necessary parameter {} is missing!".format(k))


def count_parameters(model):
  """ returns the total number of parameters of a pytorch model. """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def terminal_format(args):
  """
  args is a list of tuples and returns one string line.
  2-tuples are "{x[1]}".format(x[0]) i.e. value, format
  3-tuples are "{}={x[2]}".format(x[0],x[1]) i.e. label, value, format
  """
  line = ""
  for x in args:
    if len(x) == 3:
      line += ("{}={"+str(x[2])+"}").format(str(x[0]), x[1])
    elif len(x) == 2:
      line += ("{"+str(x[1])+"}").format(x[0])
    line += " "
  return line


def tf_add_scalars(writer, labels, scalars):
  """ Small helper function in order to perform multiple tensorboard
  write operations. """
  assert len(labels) == len(scalars)  
  global_step = scalars[0]
  for i in range(1, len(labels)):
    if np.isnan(scalars[i]).any():
      print("ERRR! Writing nan to tf log! label:{} value:{}".format(labels[i], scalars[i]))
    writer.add_scalar(labels[i], scalars[i], global_step=global_step)
  writer.flush()


def setup_log_folder(path, force=0):
  """ Creates the folders necessary for path to exist. If a folder exists and
  force=0 (default) it asks for user input. force=1 always removes it.
  force=2 always keeps it. """
  if not os.path.exists(path):
    print("creating new log directory...")
    os.makedirs(path)
    return

  # folder already exists, request user input if not forced
  print("WARNING: The results directory ({}) already exists!".format(path))
  print("Delete previous results directory [y/n]? ", end="")
  if force == 0:
    choice = input()
    while choice not in ["y", "Y", "n", "N"]:
      print("invalid answer. try again.", end="")
      choice = input()
  elif force == 1:
    choice = "y"
    print(choice)
  elif force == 2:
    choice = "N"
    print(choice)

  if choice == "y" or choice == "Y":
    print("removing directory ...")
    shutil.rmtree(path)
    print("creating new log directory...")
    os.makedirs(path)


def save_current_script(log_folder):
  """ Takes all .py files in the current folder (getcwd) and saves them in
  log_folder/source_code. Assumes log_folder exists. Does NOT copy scripts
  if the source_code folder already exists. """
  source_folder = os.getcwd()
  target_folder = os.path.join(log_folder, "source_code")
  if os.path.exists(target_folder):
    # do not copy scripts if folder alreay exists
    return
  # create target folder
  os.makedirs(target_folder)
  # recursively copy all python files
  for path, _, file_names in os.walk("."):
    # skip log_folder itself or any folder with name logs
    if path.find(log_folder) >= 0 or path.find("logs") >= 0:
      continue
    for name in file_names:
      if name[-3:] == ".py":
        src_path = os.path.join(source_folder, path)
        trg_path = os.path.join(source_folder, target_folder, path)
        if not os.path.exists(trg_path):
          os.makedirs(trg_path)
        shutil.copyfile(src=os.path.join(src_path, name),
                        dst=os.path.join(trg_path, name))


def setup_logger(log_folder, file_name="output.log"):
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.INFO)
  # create terminal handler
  s_handler = logging.StreamHandler()
  s_format = logging.Formatter('%(message)s')
  s_handler.setFormatter(s_format)
  logger.addHandler(s_handler)
  # create file handler
  if log_folder is not None:
    f_handler = logging.FileHandler(os.path.join(log_folder, file_name))
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

  return lambda *x: logger.info((x[0].replace('{', '{{').replace('}', '}}')
                                 + "{} " * (len(x)-1)).format(*x[1:])), logger


def save_config(params, path, file_name):
    # dump sacred config to file if it doesn't exist
    conf_file = os.path.join(path, file_name)
    if not os.path.exists(conf_file):
      with open(conf_file, "wb") as f:
        pickle.dump(params, f)


def load_config(path, file_name):
    # dump sacred config to file if it doesn't exist
    conf_file = os.path.join(path, file_name)
    if not os.path.exists(conf_file):
      raise FileNotFoundError
    with open(conf_file, "rb") as f:
      params = pickle.load(f)
    return params


def grow_batch(x, new_size: int):
  assert new_size >= x.shape[0], "new size has to be bigger"
  new_shape = list(x.shape)
  new_shape[0] = new_size
  new_x = torch.zeros(new_shape).to(x.device)
  new_x[:x.shape[0]] = x
  return new_x


def length_to_mask(length, max_len: int):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    Source: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
      max_len = length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype)\
                .expand(len(length), max_len) < length.unsqueeze(1)
    return mask


def traverse(tree, func, types=(list, tuple)):
  if isinstance(tree, types):
    obj = type(tree)
    tree = obj(
      [traverse(node, func, types) for node in tree]
    )
  else:
    tree = func(tree)
  return tree


def trace(self, desc, obj, active=False):
  if not active:
    return

  attr = "_trace_list"

  if len(desc) > 0:
    prefix = str(self.__class__.__name__) + "." + str(desc)
  else:
    prefix = str(self.__class__.__name__)

  # create trace list if it doesn't exist yet
  if not hasattr(self, attr):
    setattr(self, attr, [])

  trace_list = getattr(self, attr)
  # if obj has its own trace attribute, then add its content to the one of self
  if hasattr(obj, attr):
    for name, value in getattr(obj, attr):
      name = prefix + "." + name
      trace_list.append((name, value))
    # delete once read
    delattr(obj, attr)
    setattr(obj, attr, [])

  # add simply the obj as new value
  elif type(obj) == torch.Tensor:
    trace_list.append((prefix, obj.detach().cpu().numpy()))
  else:
    raise Exception("Object is of type {} and has not attribute _trace_list!".
                    format(type(obj)))


def shuffle_local(sorted_list, key, delta=10):
  """ shuffles a sorted list in place but only locally based on length """
  curr_idx = min_idx = 0
  min_len = key(sorted_list[curr_idx])
  while curr_idx < len(sorted_list):
    curr_len = key(sorted_list[curr_idx])
    if curr_len - min_len < delta:
      curr_idx += 1
    else:
      # shuffle local part of the list with similar length
      partial_list = sorted_list[min_idx:curr_idx]
      random.shuffle(partial_list)
      sorted_list[min_idx:curr_idx] = partial_list
      # set new boundaries
      min_len = curr_len
      min_idx = curr_idx


def cosine_loss(a, b):
  loss = torch.einsum("bni,bni->bn", a, b) / (a.norm(dim=-1) * b.norm(dim=-1))
  loss = (loss - 1).pow(2)
  return loss


class DummyParallel(torch.nn.Module):
  def __init__(self, module):
    super().__init__()
    self.module = module

  def forward(self, *args):
    y = self.module.forward(*args)
    trace(self, "", self.module, active=self.module.p.trace)
    return y


class CsvWriter:
  def __init__(self, column_names, path, file_name):
    self.csv_file = os.path.join(path, file_name)
    self.file = open(self.csv_file, "w+")
    self.writer = csv.writer(self.file)
    self.writer.writerow(column_names)

  def write(self, values):
    self.writer.writerow(values)
    self.file.flush()

  def close(self):
    self.file.close()


class StopWatch:
  """ Keeps track of time through pauses. """

  def __init__(self, start_with=0, start=True):
    self.total = start_with
    if start:
      self.start_time = time.time()
    else:
      self.start_time = 0

  def start(self):
    """ Sets a starting time and begins counting. """
    if self.start_time != 0:
      raise Exception("Stopwatch is already running?! ")
    self.start_time = time.time()

  def pause(self):
    """ Stops the running timer and adds it to the total time. """
    if self.start_time == 0:
      raise Exception("Stopwatch is not running. ")
    time_passed = time.time() - self.start_time
    self.total += time_passed
    self.start_time = 0

  def read(self):
    """ Outputs the current total time. """
    if self.start_time == 0:
      time_passed = 0
    else:
      time_passed = time.time() - self.start_time
    return self.total + time_passed

  def read_and_reset(self):
    """ Delete any ongoing timer or past total """
    passed = self.read()
    self.start_time = 0
    self.total = 0
    return passed


class MLP(torch.nn.Module):
  def __init__(self, dims, act):
    super().__init__()
    if len(dims) % 2 == 0:
      raise Exception("MLP dims arg must be odd.")
    self.act = act
    self.layers = []
    for i in range(len(dims)-1):
      self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
    self.layers = torch.nn.ModuleList(self.layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
      x = self.act(x)
    return x


class LazyDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def sample_mask(self, x, dropout):
        mask = x.data.new(x.shape).bernoulli_(1 - dropout)
        self.mask = Variable(mask, requires_grad=False) / (1 - dropout)

    def forward(self, x):
        if not self.training:
            return x
        bsz = x.size(0)
        mask = self.mask[:bsz].expand_as(x)
        return mask * x

class SharedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout):
        if not self.training:
            return x
        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

def embedding_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X