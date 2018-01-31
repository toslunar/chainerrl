from chainerrl.links.mlp import MLP
from chainerrl.q_function import StateQFunction


class NewLSTM(chainer.Chain):
    def __init__(self, in_size, out_size):
        super().__init__()
        with self.init_scope():
            self.nstep = chainer.links.NStepLSTM(
                n_layers=1, in_size=in_size, out_size=out_size)

        self.reset_state()

    def get_state(self):
        return self._states

    def set_state(self, state):
        self._state = state

    def reset_state(self):
        self._states = None, None

    def _init_hx(self, xs):
        x = xs[0]
        with cuda.get_device_from_id(self._device_id):
            return variable.Variable(self.xp.zeros(
                (1, len(x), self.nstep.out_size),
                dtype=xs[0].dtype))

    def __call__(self, xs):
        assert all(
            len(x0.data) >= len(x1.data)
            for x0, x1 in zip(xs, xs[1:])
        ), "inputs must be sorted in descending order of their lengths"

        hx, cx = self._state

        if hx is None:
            hx = self._init_hx(xs)

        if cx is None:
            cx = self._init_hx(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self.nstep]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self.nstep]

        hy, cy, ys = chainer.functions.n_step_lstm(
            self.nstep.n_layers, self.nstep.dropout,
            hx, cx, ws, bs, xs)

        self._state = hy, cy
        return ys


class NewFCLSTMStateQFunction(chainer.Chain, StateQFunction, Recurrent):
    """Fully-connected state-input discrete  Q-function.
    Args:
        n_dim_obs: number of dimensions of observation space
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels before LSTM
        n_hidden_layers: number of hidden layers before LSTM
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_dim_obs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        super().__init__()
        with self.init_scope():
            self.fc = MLP(in_size=self.n_input_channels,
                          out_size=n_hidden_channels,
                          hidden_sizes=[self.n_hidden_channels] *
                          self.n_hidden_layers)
            self.lstm = NewLSTM(n_hidden_channels, n_hidden_channels)
            self.out = L.Linear(n_hidden_channels, n_dim_action)

        self.reset_state()

    def get_state(self):
        return self.lstm.get_state()

    def set_state(self, state):
        self.lstm.set_state(state)

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x, batch_sizes=None):
        xp = self.xp

        h = F.relu(self.fc(x))

        if batch_sizes is None:
            h = (h,)
        else:
            sections = xp.cumsum(batch_sizes)
            assert sections[-1] == len(x)
            h = chainer.functions.split_axis(h, sections[:-1])
        h = self.lstm(h)

        return DiscreteActionValue(self.out(h))
