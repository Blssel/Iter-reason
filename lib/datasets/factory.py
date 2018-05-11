#coding:utf-8
"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.coco import coco
from datasets.ade import ade
from datasets.visual_genome import visual_genome

import numpy as np


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()
