import gc
import io
import os
import time
import sys

import numpy as np

import torch
from torch import nn
import tensorflow as tf
import logging

import torch.distributed
# Keep the import below for registering all model definitions
import losses
from model import utils as mutils
from model.ema import ExponentialMovingAverage
import datasets
from absl import flags
from torchvision import transforms as T 
# Distributed Training
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from util import save_checkpoint, restore_checkpoint,move_batch_to_device
from torch.nn.parallel import DistributedDataParallel

FLAGS = flags.FLAGS

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  

  # Initialize distributed training
  init_process_group(backend='nccl')
  
  config.local_rank = int(os.environ['LOCAL_RANK'])
  config.global_rank = int(os.environ['RANK'])
  if config.local_rank == 0:
    logging.info("Starting training process %d" % config.global_rank)
    logging.info("Config: %s" % config)


  torch.cuda.set_device(config.local_rank)
  config.device = torch.device('cuda')



  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)


  # Initialize model.
  model = mutils.create_model(config)
  # distribute model
  model = DistributedDataParallel(model, device_ids=[config.local_rank])
  
  ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, model.parameters())
  state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = dict(step=0, optimizer=optimizer,model=model, ema=ema)
  if tf.io.gfile.exists(checkpoint_meta_dir):
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  
  initial_step = int(state['step'])

  # Build pytorch dataloader for training
  #transform=T.Compose([T.ToTensor(), T.CenterCrop((config.data.image_size1, config.data.image_size2))])
  transform= T.Compose([T.ToTensor()])
  train_dl,eval_dl= datasets.create_dataloader(config,transform=transform)
  num_data = len(train_dl.dataset)
  print(f'Number of training data: {num_data}')


  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(model, train=True, optimize_fn=optimize_fn,configs=config)
  eval_step_fn = losses.get_step_fn(model, train=False, optimize_fn=optimize_fn,configs=config)


  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if config.local_rank == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))

  for epoch in range(1, config.training.epochs):
    if config.local_rank == 0:
      print('=================================================')
      print(f'Epoch: {epoch}')
      print('=================================================')

    for step, batch in enumerate(train_dl, start=1):
      batch = move_batch_to_device(batch, config.device)
      # Execute one training step
      loss = train_step_fn(state, batch)
      if step % config.training.log_freq == 0:
        if config.local_rank == 0:
          logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
          global_step = num_data * epoch + step
          writer.add_scalar("training_loss", scalar_value=loss, global_step=global_step)
      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and config.local_rank == 0:
        save_checkpoint(checkpoint_meta_dir, state)
      #Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0 and config.local_rank == 0:
        eval_batch = move_batch_to_device(next(iter(eval_dl)),config.device)
        eval_loss = eval_step_fn(state, eval_batch)
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
        global_step = num_data * epoch + step
        writer.add_scalar("eval_loss", scalar_value=eval_loss.item(), global_step=global_step)

    # Save a checkpoint for every epoch
    #save every to increase training time
    if epoch % config.training.save_every == 0 and config.local_rank == 0:
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'), state)


    #Generate and save 50um images for each epoch
    if config.training.snapshot and config.local_rank == 0:
      ema.store(model.parameters())
      ema.copy_to(model.parameters())
      eval_batch = move_batch_to_device(next(iter(eval_dl)),config.device)
      sample = model(eval_batch['hr'])
      ema.restore(model.parameters())
      this_sample_dir = os.path.join(sample_dir, "iter_{}".format(epoch))
      tf.io.gfile.makedirs(this_sample_dir)
      nrow = int(np.sqrt(sample.shape[0]))
      image_grid = make_grid(sample, nrow, padding=2)
      sample = np.clip(sample.permute(0, 2, 3, 1).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
        np.save(fout, sample)

      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
        save_image(image_grid, fout)

  #cleanup distributed training
  destroy_process_group()
