import chainer
import chainerrl
import numpy as np


qav = chainerrl.action_value.QuadraticActionValue(
    chainer.Variable(np.zeros((5, 5), dtype=np.float32)),
    chainer.Variable(np.ones((5, 5, 5), dtype=np.float32)),
    chainer.Variable(np.zeros((5, 1), dtype=np.float32)),
)

var = chainerrl.misc.collect_variables(qav)
chainerrl.misc.draw_computational_graph(var, 'tmpgraph')
