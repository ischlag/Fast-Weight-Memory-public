"""
Implements the trainer class that keeps track of the training state with some features:
- saves best and last model after every evaluation
- uses the log function to log terminal/text file
- can restore the state of a training run and continue seamlessly
- early stopping
- saves last model whenever it logs (remove if model is big!)
- saves best model whenever it evaluates
- writes logs to tf.records but also csv file
- keeps track of accuracies for individual tasks and both settings,
  LM: language modelling, RA: answer tokens only
- saves the source folder
"""

import os
import numpy as np
import torch
import copy

from tensorboardX import SummaryWriter
from munch import Munch, munchify
from utils.lib import *

BEST_MODEL_FILE_NAME = "best_eval_state.pt"
LAST_MODEL_FILE_NAME = "last_eval_state.pt"
TF_TRAIN_FOLDER_NAME = "train"
TF_EVAL_FOLDER_NAME = "valid"
CSV_FILE_NAME = "exp_logging.csv"
CONFIG_FILE_NAME = "exp_conf.pickle"
NECESSARY_PARAMS = [
  "log_every_n_steps",
  "eval_every_n_steps",
  "device",
  "log_folder",
  "max_steps",  # -1 runs indefinitely
  "write_logs",
  "early_stopping_steps",  # -1 ignores early stopping
  "QM",  # id of ra-mode positions
  "PAD",
  "ra_mode",  # identify losses for all y or just following the <ra> token
]
TRAIN_LABELS = [
  "step",
  "loss",
  "mem_norm",
  "mem_abs_max",
  "ra_loss",
  "lm_loss",
  "regularizer",
  "ra_accuracy",
  "lm_accuracy",
  "batches_per_sec",
  "tokens_per_sec",
] + ["ra_acc_task_{}".format(i) for i in range(21)] \
  + ["lm_acc_task_{}".format(i) for i in range(21)]
EVAL_LABELS = [
  "step",
  "loss",
  "mem_norm",
  "mem_abs_max",
  "ra_loss",
  "lm_loss",
  "regularizer",
  "ra_accuracy",
  "lm_accuracy",
  "batches_per_sec",
  "tokens_per_sec",
] + ["ra_acc_task_{}".format(i) for i in range(21)] \
  + ["lm_acc_task_{}".format(i) for i in range(21)]


def load_default_params(p):
  p["log_every_n_steps"] = 25
  p["eval_every_n_steps"] = 250
  p["eval_steps"] = -1
  p["log_folder"] = "logs/"
  p["max_steps"] = -1
  p["write_logs"] = True
  p["early_stopping_steps"] = -1
  p["lr_warmup"] = -1


def get_string_description(p):
  return ""


class Trainer:
  def __init__(self, model, params, train_generator, eval_generator,
               optimizer, criterion, log):
    assert_entries_exist(params, NECESSARY_PARAMS)
    self.p = params
    self.model = model.to(self.p.device)
    self.optimizer = optimizer
    self.criterion = criterion
    self.train_generator = train_generator
    self.train_iterator = iter(train_generator)
    self.eval_generator = eval_generator
    self.log = log

    # captures a restorable state
    self.state = Munch()
    self.state.global_step = 0
    self.state.epochs = 0
    self.state.train_time = 0
    self.state.total_time = 0
    self.state.hidden_state = None

    self.state.best_eval_loss = float("inf")
    self.state.best_eval_ra_loss = float("inf")
    self.state.best_eval_lm_loss = float("inf")
    self.state.best_eval_ra_acc = 0
    self.state.best_eval_lm_acc = 0
    self.state.best_train_time = 0
    self.state.best_total_time = 0
    self.state.best_step = 0
    self.state.best_hidden_state = None

    # state paths
    self.best_eval_state_path = os.path.join(self.p.log_folder,
                                             BEST_MODEL_FILE_NAME)
    self.last_eval_state_path = os.path.join(self.p.log_folder,
                                             LAST_MODEL_FILE_NAME)
    # event paths (tensorboard and csv)
    self.tb_train_path = os.path.join(self.p.log_folder, TF_TRAIN_FOLDER_NAME)
    self.tb_eval_path = os.path.join(self.p.log_folder, TF_EVAL_FOLDER_NAME)

    self.tf_train_writer = None
    self.tf_eval_writer = None
    self.csv_train_writer = None
    self.csv_eval_writer = None

    if self.p.write_logs:
      # create summary writer
      self.tf_train_writer = SummaryWriter(self.tb_train_path)
      self.tf_eval_writer = SummaryWriter(self.tb_eval_path)
      # create csv log file if nonexistent
      self.csv_train_writer = CsvWriter(column_names=TRAIN_LABELS,
                                        path=self.tb_train_path,
                                        file_name=CSV_FILE_NAME)
      self.csv_eval_writer = CsvWriter(column_names=EVAL_LABELS,
                                       path=self.tb_eval_path,
                                       file_name=CSV_FILE_NAME)
      # store sacred params
      save_config(self.p, self.p.log_folder, CONFIG_FILE_NAME)

    # continue training where the last state ended (if it exists)
    if os.path.exists(self.last_eval_state_path):
      self.log("Previous model found! Reloading last state.")
      self.load_state(path=self.last_eval_state_path)

  def _forward(self, x, y, task_id, seq_len, hidden_state, voi):
    """ Compute a train/eval forward pass and update the variables
    of interest (VOI). """
    # x: [batch_size, seq_length]
    # y: [batch_size, seq_length]
    # task_id: [batch_size, seq_length]
    # seq_len: [batch_size]

    # sort batches from longest to shortest for dynamic batch_size
    _, indecies = torch.sort(seq_len, descending=True)
    x = x[indecies]
    y = y[indecies]
    task_id = task_id[indecies]
    seq_len = seq_len[indecies]

    # move batch to accelerator
    x = x.to(self.p.device)
    y = y.to(self.p.device)
    task_id = task_id.to(self.p.device)
    seq_len = seq_len.to(self.p.device)

    # feed the model to compute the logits and loss
    logits, regularizer, hidden_state = self.model(x, seq_len, hidden_state)
    # logits: [batch_size, seq_len, vocab_size]
    # regularizer: [batch_size, 1] or None


    def compute_loss(logits, mask):
      if mask.sum().item() == 0:
        loss = torch.tensor(0).to(mask.device)
      else:
        loss = self.criterion(logits[mask], y[mask].view(-1))
      return loss

    ra_loss = compute_loss(logits, x == self.p.QM)
    lm_loss = compute_loss(logits, x != self.p.PAD)


    def compute_accs(logits, y, task_id, ra_pos):
      # y[ra_pos]: [batch_size * n_matches, 1]
      if ra_pos.sum().item() == 0:
        mean_accuracy = torch.tensor(0).long().to(ra_pos.device)
        n_per_task = torch.zeros(21).long().to(ra_pos.device)
        correct_per_task = torch.zeros(21).long().to(ra_pos.device)
      else:
        # compute VOIs
        accuracy = (torch.argmax(logits[ra_pos], dim=1) == y[ra_pos].view(-1)).int()
        # accuracy: [batch_size * n_matches]
        mean_accuracy = torch.mean(accuracy.float(), dim=0)
        # mean_accuracy: []

        # accuracy mask for individual tasks (with dim 0 for all tasks)
        # task_id: [batch_size * n_matches]
        mask = torch.stack([(i == task_id[ra_pos])
                          for i in range(0, 21)], dim=1).int()
        # mask: [batch_size * n_matches, 21]
        # keep all for task 0
        mask[:, 0] = torch.ones(mask.shape[0])

        # number of elements per task
        n_per_task = mask.sum(dim=0)
        # n_per_task: [21]

        # compute the masked accuracy
        accuracy_masked = torch.stack([accuracy]*21, dim=1) * mask
        correct_per_task = accuracy_masked.sum(dim=0)
        # correct_per_task: [21]

      return mean_accuracy, n_per_task, correct_per_task

    _, ra_n_per_task, ra_correct_per_task = compute_accs(
        logits.cpu(), y.cpu(), task_id.cpu(), x.cpu() == self.p.QM)
    _, lm_n_per_task, lm_correct_per_task = compute_accs(
        logits.cpu(), y.cpu(), task_id.cpu(), x.cpu() != self.p.PAD)

    if self.p.ra_mode:
      loss = ra_loss
    else:
      loss = lm_loss

    if regularizer is not None:
      # average across batch dimension
      regularizer = torch.mean(regularizer, dim=0)
      voi.regularizer.append(regularizer.item())
      loss = loss + self.p.regularize * regularizer

    # Note: this might not be tracked by the model
    if hasattr(self.model, "avg_mem"):
      avg_mem = self.model.avg_mem
      mem_norm = avg_mem[0].norm()
      mem_abs_max = avg_mem[0].abs().max()
    else:
      mem_norm = torch.tensor([0])
      mem_abs_max = torch.tensor([0])

    token_count = seq_len.sum()

    # track VOIs
    voi.losses.append(loss.item())
    voi.mem_norms.append(mem_norm.item())
    voi.mem_abs_maxs.append(mem_abs_max.item())

    voi.ra_losses.append(ra_loss.item())
    voi.ra_n_per_task.append(ra_n_per_task.detach())
    voi.ra_correct_per_task.append(ra_correct_per_task.detach())

    voi.lm_losses.append(lm_loss.item())
    voi.lm_n_per_task.append(lm_n_per_task.detach())
    voi.lm_correct_per_task.append(lm_correct_per_task.detach())

    voi.token_counts.append(token_count.item())

    return loss, hidden_state

  def train(self):
    self.log("Starting train ...")
    self.log("log_folder: {}\n".format(self.p.log_folder))
    self.model.train()
    self.optimizer.zero_grad()

    # variables of interest (don't forget to reset them after logging)
    train_voi = Munch()
    train_voi.losses = []
    train_voi.mem_norms = []
    train_voi.mem_abs_maxs = []

    train_voi.ra_losses = []
    train_voi.ra_accuracies = []
    train_voi.ra_n_per_task = []
    train_voi.ra_correct_per_task = []

    train_voi.lm_losses = []
    train_voi.lm_accuracies = []
    train_voi.lm_n_per_task = []
    train_voi.lm_correct_per_task = []

    train_voi.token_counts = []
    train_voi.batches = 0
    train_voi.regularizer = []

    # timers
    step_time = StopWatch(start=False)  # forward pass time
    loading_time = StopWatch()  # complement to step_time
    log_time = StopWatch()  # time passed between logs
    train_time = StopWatch(start_with=self.state.train_time)
    total_time = StopWatch(start_with=self.state.total_time)

    while self.state.global_step < self.p.max_steps or self.p.max_steps == -1:
      # get next batch and reset iterator if epoch is over
      try:
        x, y, task_id, seq_len = next(self.train_iterator)
        # x: [batch_size, seq_length]
        # y, task_id, seq_len: [batch_size, 1]
      except StopIteration:
        self.state.epochs += 1
        self.train_iterator = iter(self.train_generator)
        continue

      loading_time.pause()
      step_time.start()

      # run forward
      curr_state = self.state.hidden_state
      loss, curr_state = self._forward(
          x, y, task_id, seq_len, curr_state, train_voi
        )
      self.state.hidden_state = traverse(curr_state, lambda t: t.detach())

      # stop taining is loss is nan
      if torch.isnan(loss):
        self.log("loss is nan. Exit train().")
        return

      # update weights
      self.optimizer.zero_grad()
      if loss != 0:
        loss.backward()

        if self.p.lr_warmup > 0 and self.state.global_step < self.p.lr_warmup:
          new_lr = self.p.learning_rate * self.state.global_step / self.p.lr_warmup
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.optimizer.step()

      step_time.pause()
      loading_time.start()

      self.state.global_step += 1
      train_voi.batches += 1

      # log train summaries
      if self.state.global_step % self.p.log_every_n_steps == 0:
        # compute summaries
        ra_sum_correct = torch.stack(train_voi.ra_correct_per_task, dim=0).sum(dim=0)[0]
        ra_sum_targets = torch.stack(train_voi.ra_n_per_task, dim=0).sum(dim=0)[0]
        avg_ra_acc = ra_sum_correct.float() / ra_sum_targets.float()

        lm_sum_correct = torch.stack(train_voi.lm_correct_per_task, dim=0).sum(dim=0)[0]
        lm_sum_targets = torch.stack(train_voi.lm_n_per_task, dim=0).sum(dim=0)[0]
        avg_lm_acc = lm_sum_correct.float() / lm_sum_targets.float()

        avg_loss = np.mean(train_voi.losses)
        avg_mem_norm = np.mean(train_voi.mem_norms)
        avg_mem_abs_max = np.mean(train_voi.mem_abs_maxs)
        avg_regularizer = np.mean(train_voi.regularizer) \
                          if len(train_voi.regularizer) > 0 else 0
        avg_ra_loss = np.mean(train_voi.ra_losses)
        avg_lm_loss = np.mean(train_voi.lm_losses)
        sum_token_count = np.sum(train_voi.token_counts)
        avg_loading_time = loading_time.read() / train_voi.batches
        avg_step_time = step_time.read() / train_voi.batches

        secs_passed = log_time.read_and_reset()
        hours_passed = total_time.read() / (60.0 * 60.0)
        batches_per_sec = train_voi.batches / secs_passed
        tokens_per_sec = sum_token_count / secs_passed

        # keep track of train and total time
        self.state.train_time = train_time.read()
        self.state.total_time = total_time.read()

        # write terminal and file summaries
        vars = [
          ("ep", self.state.epochs, ""),
          ("step", self.state.global_step, ":4"),
          ("loss", avg_loss, ":.5f"),
          ("reg", avg_regularizer, ":.5f"),
          ("ra_acc", avg_ra_acc * 100, ":5.2f"),
          ("lm_acc", avg_lm_acc * 100, ":5.2f"),
          ("ra_loss", avg_ra_loss, ":.5f"),
          ("lm_loss", avg_lm_loss, ":.5f"),
          ("hours", hours_passed, ":.2f"),
          ("b/s", batches_per_sec, ":.2f"),
          ("t/s", tokens_per_sec, ":5.0f"),
        ]
        self.log(terminal_format(vars))

        # write tensorboard and csv summaries
        if self.p.write_logs:
          scalars = [self.state.global_step,
                     avg_loss,
                     avg_mem_norm,
                     avg_mem_abs_max,
                     avg_ra_loss,
                     avg_lm_loss,
                     avg_regularizer,
                     avg_ra_acc,
                     avg_lm_acc,
                     batches_per_sec,
                     tokens_per_sec]
          ra_correct_stack = torch.stack(train_voi.ra_correct_per_task).sum(0).float()
          ra_count_stack = torch.stack(train_voi.ra_n_per_task).sum(0).float()
          # ra_correct_stack, ra_count_stack: [21]

          ra_mask_nonzero = ra_count_stack > 0.0
          # ra_nonzero_batches: [21]

          ra_accs = torch.zeros_like(ra_correct_stack).float().to(ra_correct_stack.device)
          ra_accs[ra_mask_nonzero] = ra_correct_stack[ra_mask_nonzero] / ra_count_stack[ra_mask_nonzero]
          # ra_accs: [21]
          scalars += [acc.item() for acc in ra_accs]


          lm_correct_stack = torch.stack(train_voi.lm_correct_per_task).sum(0).float()
          lm_count_stack = torch.stack(train_voi.lm_n_per_task).sum(0).float()
          # lm_correct_stack, lm_count_stack: [21]

          lm_mask_nonzero = lm_count_stack > 0
          # lm_mask_nonzero: [21]

          lm_accs = torch.zeros_like(lm_count_stack).float().to(lm_correct_stack.device)
          lm_accs[lm_mask_nonzero] = lm_correct_stack[lm_mask_nonzero] / lm_count_stack[lm_mask_nonzero]
          # ra_accs: [21]
          scalars += [acc.item() for acc in lm_accs]

          tf_add_scalars(self.tf_train_writer, TRAIN_LABELS, scalars)
          self.csv_train_writer.write(scalars)
          # restarts mess a little with tensorboard, saving the state here
          # would help to deal with that but it is a slow down for big models.
          # self.save_state(target=self.last_eval_state_path)

        # clear
        log_time.start()
        step_time.read_and_reset()
        loading_time.read_and_reset()
        loading_time.start()
        train_voi.losses = []
        train_voi.mem_norms = []
        train_voi.mem_abs_maxs = []

        train_voi.ra_losses = []
        train_voi.ra_accuracies = []
        train_voi.ra_n_per_task = []
        train_voi.ra_correct_per_task = []

        train_voi.lm_losses = []
        train_voi.lm_accuracies = []
        train_voi.lm_n_per_task = []
        train_voi.lm_correct_per_task = []

        train_voi.token_counts = []
        train_voi.batches = 0
        train_voi.regularizer = []

      # run evaluation
      if self.state.global_step % self.p.eval_every_n_steps == 0:
        loading_time.pause()
        log_time.pause()
        train_time.pause()

        # build the initial hidden state for evaluation
        # replicate the hidden state of the first row in the train_batch
        pick_first = lambda t: t[[0]]
        stack_states = lambda t: torch.cat([t]*self.p.eval_batch_size, dim=0)
        h = copy.deepcopy(self.state.hidden_state)
        single_hidden_state = traverse(h, pick_first)
        eval_hidden_state = traverse(single_hidden_state, stack_states)

        self.evaluate(write_logs=self.p.write_logs,
                      hidden_state=eval_hidden_state)
        self.model.train()

        # check for early stopping
        steps_without_progress = self.state.global_step - self.state.best_step
        if self.p.early_stopping_steps >= 0 and \
           steps_without_progress > self.p.early_stopping_steps:
          self.log("No progress for {} steps".format(steps_without_progress))
          self.log("Stopping training.")
          return

        loading_time.start()
        log_time.start()
        train_time.start()

  def evaluate(self, generator=None, write_logs=False, hidden_state=None, progress=False):
    if generator is None:
      generator = self.eval_generator
    n_samples = len(generator)

    self.model.eval()

    # variables of interest
    eval_voi = Munch()
    eval_voi.losses = []
    eval_voi.mem_norms = []
    eval_voi.mem_abs_maxs = []

    eval_voi.ra_losses = []
    eval_voi.ra_accuracies = []
    eval_voi.ra_n_per_task = []
    eval_voi.ra_correct_per_task = []

    eval_voi.lm_losses = []
    eval_voi.lm_accuracies = []
    eval_voi.lm_n_per_task = []
    eval_voi.lm_correct_per_task = []

    eval_voi.token_counts = []
    eval_voi.batches = 0
    eval_voi.regularizer = []

    # timers
    step_time = StopWatch(start=False)  # forward pass time
    loading_time = StopWatch()  # complement to step_time
    eval_time = StopWatch()

    with torch.no_grad():
      counter = 0
      start_time = time.time()
      for x, y, task_id, seq_len in generator:      

        # forward pass and track variables
        loading_time.pause()
        step_time.start()

        _, hidden_state = self._forward(x, y, task_id, seq_len,
                                        hidden_state, eval_voi)

        step_time.pause()
        loading_time.start()
        eval_voi.batches += 1

        # print progress
        if progress and counter % 100 == 0 and counter > 0:
          elapsed = time.time() - start_time
          speed = elapsed / counter
          remaining = (n_samples - counter) * speed / 60.
          print("{}/{} done. ~{:.1f} mins remaining".format(counter, n_samples, remaining))
        counter += 1

        if self.p.eval_steps > 0 and eval_voi.batches > self.p.eval_steps:
          break

    loading_time.pause()

    # compute summaries
    ra_sum_correct = torch.stack(eval_voi.ra_correct_per_task, dim=0).sum(dim=0)[0]
    ra_sum_targets = torch.stack(eval_voi.ra_n_per_task, dim=0).sum(dim=0)[0]
    avg_ra_acc = ra_sum_correct.float() / ra_sum_targets.float()

    lm_sum_correct = torch.stack(eval_voi.lm_correct_per_task, dim=0).sum(dim=0)[0]
    lm_sum_targets = torch.stack(eval_voi.lm_n_per_task, dim=0).sum(dim=0)[0]
    avg_lm_acc = lm_sum_correct.float() / lm_sum_targets.float()

    avg_loss = np.mean(eval_voi.losses)
    avg_mem_norm = np.mean(eval_voi.mem_norms)
    avg_mem_abs_max = np.mean(eval_voi.mem_abs_maxs)
    avg_ra_loss = np.mean(eval_voi.ra_losses)
    avg_lm_loss = np.mean(eval_voi.lm_losses)
    avg_regularizer = np.mean(eval_voi.regularizer) \
                      if len(eval_voi.regularizer) > 0 else 0
    sum_token_count = np.sum(eval_voi.token_counts)

    secs_passed = eval_time.read_and_reset()
    batches_per_sec = eval_voi.batches / secs_passed
    tokens_per_sec = sum_token_count / secs_passed

    # track best summaries so far and save state/model
    if avg_loss < self.state.best_eval_loss and write_logs:
      # new best model
      self.state.best_eval_loss = avg_loss
      self.state.best_eval_ra_loss = avg_ra_loss
      self.state.best_eval_lm_loss = avg_lm_loss
      self.state.best_eval_ra_acc = avg_ra_acc
      self.state.best_eval_lm_acc = avg_lm_acc
      self.state.best_train_time = self.state.train_time
      self.state.best_total_time = self.state.total_time
      self.state.best_step = self.state.global_step
      # save best state so far
      self.save_state(target=self.best_eval_state_path)

    # save current state
    if write_logs:
      self.save_state(target=self.last_eval_state_path)

    # write terminal and file summaries
    vars = [
      ("eval", ""),
      ("loss", avg_loss, ":.5f"),
      ("reg", avg_regularizer, ":.5f"),
      ("ra_acc", avg_ra_acc * 100, ":5.2f"),
      ("lm_acc", avg_lm_acc * 100, ":5.2f"),
      ("ra_loss", avg_ra_loss, ":.5f"),
      ("lm_loss", avg_lm_loss, ":.5f"),
      ("b/s", batches_per_sec, ":.2f"),
      ("t/s", tokens_per_sec, ":5.0f"),
      ("| best:", ""),
      ("loss", self.state.best_eval_loss, ":.5f"),
      ("ra_acc", self.state.best_eval_ra_acc * 100, ":5.2f"),
      ("lm_acc", self.state.best_eval_lm_acc * 100, ":5.2f"),
    ]
    self.log("")
    self.log(terminal_format(vars))
    # print folder path for easier identification of running experiments
    self.log("(" + self.p.log_folder + ")")
    self.log("")

    # write tensorboard summaries
    if write_logs:
      scalars = [self.state.global_step,
                 avg_loss,
                 avg_mem_norm,
                 avg_mem_abs_max,
                 avg_ra_loss,
                 avg_lm_loss,
                 avg_regularizer,
                 avg_ra_acc,
                 avg_lm_acc,
                 batches_per_sec,
                 tokens_per_sec]

      ra_correct_stack = torch.stack(eval_voi.ra_correct_per_task).sum(0).float()
      ra_count_stack = torch.stack(eval_voi.ra_n_per_task).sum(0).float()
      # ra_correct_stack, ra_count_stack: [21]

      ra_mask_nonzero = ra_count_stack > 0.0
      # ra_nonzero_batches: [21]

      ra_accs = torch.zeros_like(ra_correct_stack).float().to(ra_correct_stack.device)
      ra_accs[ra_mask_nonzero] = ra_correct_stack[ra_mask_nonzero] / ra_count_stack[ra_mask_nonzero]
      # ra_accs: [21]
      scalars += [acc.item() for acc in ra_accs]


      lm_correct_stack = torch.stack(eval_voi.lm_correct_per_task).sum(0).float()
      lm_count_stack = torch.stack(eval_voi.lm_n_per_task).sum(0).float()
      # lm_correct_stack, lm_count_stack: [21]

      lm_mask_nonzero = lm_count_stack > 0
      # lm_mask_nonzero: [21]

      lm_accs = torch.zeros_like(lm_count_stack).float().to(lm_correct_stack.device)
      lm_accs[lm_mask_nonzero] = lm_correct_stack[lm_mask_nonzero] / lm_count_stack[lm_mask_nonzero]
      # ra_accs: [21]
      scalars += [acc.item() for acc in lm_accs]

      tf_add_scalars(self.tf_eval_writer, EVAL_LABELS, scalars)
      self.csv_eval_writer.write(scalars)


  def save_state(self, target):
    curr_state = {
      "state": self.state,
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict()
    }
    torch.save(obj=curr_state, f=target)

  def load_state(self, path=None):
    if path is None:
      path = self.best_eval_state_path
    curr_state = torch.load(path)

    # lstm weight drop fix
    if self.p.model_name == "lm_lstm":
      for key in copy.deepcopy(curr_state["model"]).keys():
        if "old_module.W_hh" in key[-15:]:
          del curr_state["model"][key]

    self.model.load_state_dict(curr_state["model"])
    self.optimizer.load_state_dict(curr_state["optimizer"])
    self.state = munchify(curr_state["state"])
