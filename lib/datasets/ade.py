#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
try:
  import cPickle as pickle
except ImportError:
  import pickle
import json
import cv2
import numpy as np

from datasets.imdb import imdb
from model.config import cfg

class ade(imdb):
  def __init__(self, image_set, count=5):
    imdb.__init__(self, 'ade_%s_%d' % (image_set,count))
    self._image_set = image_set
    self._root_path = osp.join(cfg.DATA_DIR, 'ADE')
    self._name_file = osp.join(self._root_path, 'objectnames.txt')
    self._count_file = osp.join(self._root_path, 'objectcounts.txt')
    self._anno_file = osp.join(self._root_path, self._image_set + '.txt')
    with open(self._anno_file) as fid:
      image_index = fid.readlines()
      self._image_index = [ii.strip() for ii in image_index]
    with open(self._name_file) as fid:
      raw_names = fid.readlines()
      self._raw_names = [n.strip().replace(' ', '_') for n in raw_names]
      self._len_raw = len(self._raw_names)
    with open(self._count_file) as fid:
      raw_counts = fid.readlines()
      self._raw_counts = np.array([int(n.strip()) for n in raw_counts])

    # First class is always background
    self._ade_inds = [0] + list(np.where(self._raw_counts >= count)[0])
    self._classes = ['__background__']

    for idx in self._ade_inds:
      if idx == 0:
        continue
      ade_name = self._raw_names[idx]
      self._classes.append(ade_name)

    self._classes = tuple(self._classes)
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
