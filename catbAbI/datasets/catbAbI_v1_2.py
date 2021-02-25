import importlib
import os
import pickle
import random
import torch

from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

catbAbI10k_TEMPLATE = "en-valid-10k_{}.txt"
catbAbI1k_TEMPLATE = "en-valid_{}.txt"
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = "data/catbAbI_v1.2"
PAD_STR = "<pad>"
END_OF_STORY = "<eos>"
LARGE_10K = "catbAbI10k"
SMALL_1K = "catbAbI1k"


def load_default_params(p):
    p["dataset_variation"] = LARGE_10K
    p["num_workers"] = 2
    # Train on selected tasks by "t" e.g. "3t6t5t12" to train  on 2, 6, 5, and 12
    # empty string trains on all tasks
    p["whitelist"] = ""  
    # request answer mode -> only predict answers otherwise everything
    p["ra_mode"] = True
    p["seq_len"] = 200


def get_string_description(p):
  txt = "{}_{}{}"
  if len(p["whitelist"]) == 0:
    return txt.format(p["dataset_variation"],
                      "raMode" if p["ra_mode"] else "lmMode",
                      f"_sl{p['seq_len']}",
                      "")
  else:
    whitelist_str = "_task" + "-".join(sorted(set(p["whitelist"].split("t"))))
    return txt.format(p["dataset_variation"],
                      "raMode" if p["ra_mode"] else "lmMode",
                      f"_sl{p['seq_len']}",
                      whitelist_str)


def create_iterator(p, partition, batch_size, random=True):
  is_large = p["dataset_variation"] == LARGE_10K
  dataset = catbAbI(partition=partition,
                    whitelist=p["whitelist"],
                    ra_mode=p["ra_mode"],
                    large=is_large)
  p.PAD = dataset.word2idx[PAD_STR]
  p.EOS = dataset.word2idx[END_OF_STORY]
  p.QM = dataset.word2idx["?"]

  # create data loader
  if random:
    sampler = RandomSampler(dataset, replacement=False)
  else:
    sampler = SequentialSampler(dataset)
  batch_generator = StoryBatcher(sampler,
                                 batch_size=batch_size,
                                 seq_len=p["seq_len"],
                                 PAD=p.PAD)
  return batch_generator


def read_samples(file_path, word2idx, whitelist):
  samples = []
  with open(file_path, "r") as f:
    for line in f:
      task, story = line.rstrip('\n').split("\t")
      if str(task) in whitelist.split("t") or len(whitelist) == 0:
        words = story.split(" ")
        # encode samples
        EOS = word2idx[END_OF_STORY]
        x = [EOS] + [word2idx[word] for word in words]
        y = [word2idx[word] for word in words] + [EOS]
        t = [int(task)] * len(x)
        samples.append((x, y, t))
  return samples


class catbAbI(data.Dataset):
  def __init__(self, partition, whitelist,
               ra_mode, large=True, folder=DATA_PATH):
    self.partition = partition
    self.whitelist = whitelist
    self.ra_mode = ra_mode

    if large:
      self.fp = os.path.join(folder, catbAbI10k_TEMPLATE.format(partition))
    else:
      self.fp = os.path.join(folder, catbAbI1k_TEMPLATE.format(partition))

    with open(os.path.join(folder, "vocab.pkl"), "rb") as f:
      self.word2idx, self.idx2word = pickle.load(f)

    self.samples = read_samples(self.fp, self.word2idx, self.whitelist)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    x, y, t = self.samples[index]
    x = torch.tensor(x).long()
    y = torch.tensor(y).long()
    t = torch.tensor(t).long()

    if self.ra_mode:
      qm_pos = x != self.word2idx["?"]
      y[qm_pos] = self.word2idx[PAD_STR]

    return x, y, t, len(x)


class StoryBatcher:
  def __init__(self, sampler, batch_size, seq_len, PAD, buffer_size=None):
    super().__init__()
    self.sampler = sampler
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.PAD = PAD
    self.dataset = sampler.data_source
    if buffer_size:
      self.buffer_size = buffer_size
    else:
      self.buffer_size = seq_len * 4
    self.buffer_x = [torch.tensor([]).long()] * batch_size
    self.buffer_y = [torch.tensor([]).long()] * batch_size
    self.buffer_t = [torch.tensor([]).long()] * batch_size

  def __iter__(self):
    self.sampler_iter = iter(self.sampler)
    return self

  def __next__(self):
    # fill buffer
    while True:
      lengths = [len(t) for t in self.buffer_x]
      min_len_idx = lengths.index(min(lengths))

      if min(lengths) >= self.buffer_size:
        break

      idx = next(self.sampler_iter, None)
      if idx is None:
        break
      else:
        x, y, t, length = self.sampler.data_source[idx]
        self.buffer_x[min_len_idx] = torch.cat([self.buffer_x[min_len_idx], x])
        self.buffer_y[min_len_idx] = torch.cat([self.buffer_y[min_len_idx], y])
        self.buffer_t[min_len_idx] = torch.cat([self.buffer_t[min_len_idx], t])

      # lengths = [len(t) for t in self.buffer_x]
      # print("lengths: ", lengths)

    if sum(lengths) == 0:
      raise StopIteration

    # get a batch
    batch_x = [b[:self.seq_len] for b in self.buffer_x]
    batch_y = [b[:self.seq_len] for b in self.buffer_y]
    batch_t = [b[:self.seq_len] for b in self.buffer_t]
    batch_len = torch.tensor([len(x) for x in batch_x])

    # pop from buffer
    self.buffer_x = [b[self.seq_len:] for b in self.buffer_x]
    self.buffer_y = [b[self.seq_len:] for b in self.buffer_y]
    self.buffer_t = [b[self.seq_len:] for b in self.buffer_t]

    # pad into tensor
    x_pad = pad_sequence(batch_x, batch_first=True, padding_value=self.PAD)
    y_pad = pad_sequence(batch_y, batch_first=True, padding_value=self.PAD)
    t_pad = pad_sequence(batch_t, batch_first=True, padding_value=self.PAD)

    return x_pad, y_pad, t_pad, batch_len

  def __len__(self):
    # approximate
    words = sum([len(sample[0]) for sample in self.sampler.data_source])
    return round(words / (self.batch_size * self.seq_len) + 0.5)


if __name__ == "__main__":
  skip = random.randint(0,20)
  # ----------  draw one sample ----------
  print("\n\n######### Dataset demo ra_mode=False")
  d = catbAbI(folder="data/catbAbI_v1.2/",
              partition="train",
              whitelist="",
              ra_mode=False)

  it = iter(d)
  for _ in range(1, skip):
    batch = next(it)
  x, y, task_id = batch[0], batch[1], batch[2]

  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)

  print(x)
  print(y)
  print(task_id)
  print(" ".join([d.idx2word[x.item()] for x in batch[0]]))

  # ----------  draw one sample ----------
  print("\n\n######### Dataset demo ra_mode=True")
  d = catbAbI(folder="data/catbAbI_v1.2/",
              partition="train",
              whitelist="",
              ra_mode=True)

  it = iter(d)
  for _ in range(1, skip):
    batch = next(it)
  x, y, task_id = batch[0], batch[1], batch[2]

  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)

  print(x)
  print(y)
  print(task_id)
  print(" ".join([d.idx2word[x.item()] for x in batch[0]]))

  # ----------  draw a batch ----------
  print("\n\n######### StoryBatcher Demo")
  dataset = catbAbI(folder="data/catbAbI_v1.2/",
              partition="train",
              whitelist="",
              ra_mode=True)
  random_sampler = RandomSampler(dataset, replacement=False)
  batch_generator = StoryBatcher(random_sampler,
                                 batch_size=64,
                                 seq_len=200,
                                 PAD=0)
  it = iter(batch_generator)
  row=0
  
  x, y, task_id, seq_len = next(it)
  print("Batch 1")
  print("seq_len: ", seq_len)
  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)
  print("row 0:")
  print(x[row])
  print(y[row])
  print(" ".join([dataset.idx2word[w.item()] for w in x[row]]))

  x, y, task_id, seq_len = next(it)
  print("Batch 2")
  print("seq_len: ", seq_len)
  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)
  print("row 0:")
  print(x[row])
  print(y[row])
  print(" ".join([dataset.idx2word[w.item()] for w in x[row]]))

  # ----------  the last batch ----------
  print("\n\n######### Last batch is shorter")
  dataset = catbAbI(folder="data/catbAbI_v1.2/",
              partition="train",
              whitelist="",
              ra_mode=True)
  random_sampler = RandomSampler(dataset, replacement=False)
  batch_generator = StoryBatcher(random_sampler,
                                 batch_size=64,
                                 seq_len=200,
                                 PAD=0)
  it = iter(batch_generator)
  prev_last_batch = next(it)
  last_batch = next(it)
  count = 2
  while True:
    try:
      batch = next(it)
      prev_last_batch = last_batch
      last_batch = batch
      count += 1
    except StopIteration:
      break

  row = torch.where(last_batch[-1] > 0)[0][0]

  x, y, task_id, seq_len = prev_last_batch
  print("batch number: ", count-1)
  print("last batch")
  print("seq_len: ", seq_len)
  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)
  print("row 0:")
  print(x[row])
  print(y[row])

  x, y, task_id, seq_len = last_batch
  print("batch number: ", count)
  print("last batch")
  print("seq_len: ", seq_len)
  print("x: ", x.shape)
  print("y: ", y.shape)
  print("task_id: ", task_id.shape)
  print("row 0:")
  print(x[row])
  print(y[row])





