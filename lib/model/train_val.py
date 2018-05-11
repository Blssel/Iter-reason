#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class SolverWrapper(object):
  """
    A wrapper class for the training process
  """
  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model
    

  def get_variables_in_checkpoint_file(self, file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")


  def construct_graph(self, sess):
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default')
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.RATE, trainable=False)
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      grad_summaries = []
      for grad, var in gvs:
        if grad == None:
          continue
        grad_summaries.append(tf.summary.histogram('TRAIN/' + var.name, var))
        grad_summaries.append(tf.summary.histogram('GRAD/' + var.name, grad))

      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)
      self.summary_grads = tf.summary.merge(grad_summaries)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

      return lr, train_op


  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    
    # Get the snapshot name in TensorFlow
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    if len(sfiles) > 1:
      sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]
    else:
      sfiles = [ss.replace('.meta', '') for ss in sfiles]
    
    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
    if len(nfiles) > 1:
      nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles


  def initialize(self, sess):
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
    
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    self.net.fix_variables(sess, self.pretrained_model)
    self.net.fixed_parameters(sess)
    print('Fixed.')
    rate = cfg.TRAIN.RATE
    last_snapshot_iter = 0
    stepsizes = list(cfg.TRAIN.STEPSIZE)
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
  
    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths


  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = self.imdb.data_layer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = self.imdb.data_layer(self.valroidb, self.imdb.num_classes, random=True)

    #------------构建计算图------------#
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()
  
    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                            str(sfiles[-1]), 
                                                                            str(nfiles[-1]))
    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_iter = iter
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()

      now = time.time()
      if iter == 1 or \
          (now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL and \
          iter - last_summary_iter > cfg.TRAIN.SUMMARY_ITERS):
        # Compute the graph with summary
        loss_cls, total_loss, summary, gsummary = \
          self.net.train_step_with_summary(sess, blobs, train_op, self.summary_grads)
        self.writer.add_summary(summary, float(iter))
        self.writer.add_summary(gsummary, float(iter+1))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_iter = iter
        last_summary_time = now
      else:
        # Compute the graph without summary
        loss_cls, total_loss = self.net.train_step(sess, blobs, train_op)
      timer.toc()




















def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  for i in range(len(imdb.image_index)):
    imdb.roidb[i]['image'] = imdb.image_path_at(i)

  return imdb.roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=None,
            max_iters=40000):
"""Train a Faster R-CNN network."""
roidb = filter_roidb(roidb)
valroidb = filter_roidb(valroidb)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

with tf.Session(config=tfconfig) as sess:
  sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model)
  print('Solving...')
  sw.train_model(sess, max_iters)
  print('done solving')
