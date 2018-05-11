#coding:utf-8
class resnetv1(Network):
  def __init__(self,num_layers=50):  # 层数可变
    
    self._scope='resnet_v1_%d'%num_layers



  def _build_base(self):
    with tf.variable_scope(self._scope, self._scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net
