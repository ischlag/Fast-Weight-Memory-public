import importlib
import random
import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np

from sacred import Experiment
from munch import munchify

from utils.lib import setup_log_folder, save_current_script, setup_logger, \
  count_parameters

MODELS = "models"
TRAINERS = "trainers"
DATASETS = "datasets"

ex = Experiment("experiment")

@ex.config
def main_config():
  # general
  p = {}
  p["device"] = torch.device("cuda")
  p["n_gpus"] = torch.cuda.device_count()
  p["run"] = ""
  p["jit"] = False
  p["seed"] = random.randint(0, 100000)
  p["force"] = 0  # ask: 0, always remove: 1, always keep: 2
  p["eval_only"] = False
  p["write_logs"] = True
  p["eval_state_path"] = None

  # dataset
  p["dataset_name"] = "bAbI_v1_2"
  p["dataset_variation"] = "bAbI10k"
  dataset = importlib.import_module(DATASETS + "." + p["dataset_name"])
  dataset.load_default_params(p)
  dataset_desc = dataset.get_string_description(p)

  # model
  p["model_name"] = "seq_classification_lstm_custom"
  model = importlib.import_module(MODELS + "." + p["model_name"])
  model.load_default_params(p)
  model_desc = model.get_string_description(p)

  # trainer
  p["trainer_name"] = "bAbI_trainer"
  trainer = importlib.import_module(TRAINERS + "." + p["trainer_name"])
  trainer.load_default_params(p)
  trainer_desc = trainer.get_string_description(p)
  # since trainer desc is currently empty it is not used below

  # default parameters
  p["train_batch_size"] = 32
  p["eval_batch_size"] = 128
  p["learning_rate"] = 1e-3
  p["regularize"] = 0.0
  p["beta1"] = 0.9
  p["beta2"] = 0.999

  # build folder name
  p["log_root"] = "logs"
  path = "{}/{}/{}_lr{}_reg{}_bs{}_gpus{}{}{}/{}"
  args = [
    p["log_root"],
    dataset_desc,
    model_desc,
    p["learning_rate"],
    p["regularize"],
    p["train_batch_size"],
    p["n_gpus"],
    "_jit" if p["jit"] else "",
    "_{}".format(p["run"]) if p["run"] != "" else "",
    p["seed"]
  ]
  p["log_folder"] = path.format(*args)


@ex.automain
def run(p, _log):
  p = munchify(p)
  torch.manual_seed(p.seed)
  np.random.seed(p.seed)
  random.seed(p.seed)
  # setup log folder and backup source code
  if p.write_logs:
    setup_log_folder(p.log_folder, p.force)
    save_current_script(p.log_folder)

  # setup logger
  log, logger = setup_logger(p.log_folder if p.write_logs else None)
  log("{}".format(p))
  ex.logger = logger

  # import dataset
  log("load datasets ...")
  _module = importlib.import_module(DATASETS + "." + p.dataset_name)
  train_generator = _module.create_iterator(p=p,
                                            partition="train",
                                            batch_size=p.train_batch_size)
  eval_generator = _module.create_iterator(p=p,
                                           partition="valid",
                                           batch_size=p.eval_batch_size,
                                           random=False)
  test_generator = _module.create_iterator(p=p,
                                          partition="test",
                                          batch_size=1,
                                          random=False)

  vocab_size = len(train_generator.dataset.idx2word)
  log("dataset vocab size: {}".format(vocab_size))
  log("Number of train batches: {}".format(len(train_generator)))
  log("Number of test batches: {}".format(len(eval_generator)))

  # build model
  log("load model ...")
  _module = importlib.import_module(MODELS + "." + p.model_name)
  p.vocab_size = vocab_size
  model = _module.Model(p)
  if p.jit:
    log("compiling model with jit.script ...")
    model = jit.script(model)
  log("skipping model print ...")
  #log("{}".format(model))
  log("{} trainable parameters found. ".format(count_parameters(model)))

  # optimizer
  optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=p.learning_rate,
                               betas=(p.beta1, p.beta2))

  # loss
  criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

  # DataParallel over multiple GPUs
  if p.n_gpus > 1:
    if p.jit:
      raise Exception("JIT is currently not supported for distributed training!")
    log("{} GPUs detected. Using nn.DataParallel. Batch-size per GPU: {}"
        .format(p.n_gpus, p.train_batch_size // p.n_gpus))
    model = nn.DataParallel(model)


  # create trainer
  log("load trainer ...")
  _module = importlib.import_module(TRAINERS + "." + p.trainer_name)
  trainer = _module.Trainer(model=model,
                            params=p,
                            train_generator=train_generator,
                            eval_generator=eval_generator,
                            optimizer=optimizer,
                            criterion=criterion,
                            log=log)

  # begin training
  trainer.train()

  log("\nloading best mode from: ", trainer.best_eval_state_path)
  trainer.load_state(trainer.best_eval_state_path)

  log("\nfinal batch_size=1 evaluation ...")
  trainer.evaluate(generator=test_generator, progress=True)
