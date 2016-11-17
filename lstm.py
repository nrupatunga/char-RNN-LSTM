# This is the implementation of single layer LSTM
# ----------
# Equations:
# ----------
#   g(t) = tanh(Wgx * x(t), Wgh * h(t - 1) + bg)
#   i(t) = sigmoid(Wix * x(t), Wih * h(t - 1) + bg) ---> input Gate
#   f(t) = sigmoid(Wfx * x(t), Wfh * h(t - 1) + bg) ---> forget Gate
#   o(t) = sigmoid(Wox * x(t), Woh * h(t - 1) + bg) ---> output Gate
#   S(t) = g(t) .* i(t) + f(t) .* S(t - 1)  ---> update Cell State s(t)
#   h(t) = S(t) .* o(t) ---> update hidden state
#
# ---------
# Notation:
# ---------
#    * : dot product
#   .* : hadamard product
#
# In implementation we concatenate the weights as below
# Wg = [Wgx Wgh]
# Wi = [Wix Wih]
# Wf = [Wfx Wfh]
# Wo = [Wox Woh]
# Xs(t) = [x(t) h(t-1)]
# ----------------------------------------------------------------------
import numpy as np
from random import uniform

initial_seed = 42


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class LSTM_state:
    ''' Class holding all the states of LSTM unit '''

    def __init__(self, input_dim, output_dim, num_mem_cells):
        '''At time t=0 initialize the states to zeros'''

        self.g = np.zeros(num_mem_cells)
        self.i = np.zeros(num_mem_cells)
        self.f = np.zeros(num_mem_cells)
        self.o = np.zeros(num_mem_cells)
        self.s = np.zeros(num_mem_cells)
        self.h = np.zeros(num_mem_cells)
        self.y = np.zeros(output_dim)
        self.prob = np.zeros(output_dim)


class LSTM:

    def __init__(self, input_txt_file):
        ''' Process the input data, prepare for training
        @params:
        --------
            input_txt_file - text file input containing the test characters
        '''

        data = []
        with open(input_txt_file, 'r') as f:
            data = f.read()

        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {c: i for i, c in enumerate(chars)}
        self.index_to_char = {i: c for i, c in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.data_len = len(data)
        self.seq_len = 25
        self.objLstmNet = LSTM_network(self.vocab_size, self.vocab_size, self.seq_len)

    def train(self):
        ''' Training the LSTM'''

        ptr, sample_n = 0, 0
        objLstmNet = self.objLstmNet

        while True:
            if ptr + self.seq_len >= self.data_len or sample_n is 0:
                h_prev, s_prev = None, None
                ptr = 0

            inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
            targets = [self.char_to_index[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]
            x_list, y_list = inputs, targets
            num_samples = len(inputs)

            for i in range(num_samples):
                x_one_hot = np.zeros((self.vocab_size))
                x_one_hot[x_list[i]] = 1
                objLstmNet.feed_forward(x_one_hot, s_prev, h_prev)

            # if sample_n == 40:
                # gradient_check(inputs, targets, s_prev, h_prev)

            s_prev = objLstmNet.lstm_node_list[num_samples - 1].state.s
            h_prev = objLstmNet.lstm_node_list[num_samples - 1].state.h

            objLstmNet.calculate_loss(y_list)
            objLstmNet.feed_backward(y_list)
            objLstmNet.apply_grad()
            objLstmNet.reset()

            if sample_n % 50 == 0:
                print('Iteration {}, loss = {}'.format(sample_n, self.objLstmNet.loss))

            ptr = ptr + self.seq_len
            sample_n = sample_n + 1


class LSTM_grad_data:

    def __init__(self, input_txt_file):
        ''' Process the input data
        @params:
        --------
            input_txt_file - text file input containing the test characters
        '''

        data = []
        with open(input_txt_file, 'r') as f:
            data = f.read()

        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {c: i for i, c in enumerate(chars)}
        self.index_to_char = {i: c for i, c in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.data_len = len(data)
        self.seq_len = 25
        ptr = 0
        self.inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
        self.targets = [self.char_to_index[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]


class LSTM_param:
    ''' Class holding all the LSTM learnable weights and biases of LSTM unit'''

    def random_array(self, mu, sigma, *shape_args):
        return np.random.randn(*shape_args) * sigma + mu

    def init_param(self, num_mem_cells, output_dim, concat_dim, mu=0, sigma=0.01):
        '''initialize the weights'''

        np.random.seed(initial_seed)
        # Weight initialization
        self.wg = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wi = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wf = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wo = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wy = self.random_array(mu, sigma, output_dim, num_mem_cells)

        # Bias initialization
        self.bg = np.zeros(num_mem_cells)
        self.bi = np.zeros(num_mem_cells)
        self.bf = np.zeros(num_mem_cells)
        self.bo = np.zeros(num_mem_cells)
        self.by = np.zeros(output_dim)

        # weight gradient initialization
        self.dwg = np.zeros_like(self.wg)
        self.dwi = np.zeros_like(self.wi)
        self.dwf = np.zeros_like(self.wf)
        self.dwo = np.zeros_like(self.wo)
        self.dwy = np.zeros_like(self.wy)

        # bias gradient initialization
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)

        # memory variables for adagrad
        self.mwg = np.zeros_like(self.wg)
        self.mwi = np.zeros_like(self.wi)
        self.mwf = np.zeros_like(self.wf)
        self.mwo = np.zeros_like(self.wo)
        self.mwy = np.zeros_like(self.wy)

        self.mbg = np.zeros_like(self.bg)
        self.mbi = np.zeros_like(self.bi)
        self.mbf = np.zeros_like(self.bf)
        self.mbo = np.zeros_like(self.bo)
        self.mby = np.zeros_like(self.by)

    def apply_grad(self, loss, smooth_loss):
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        dwg = self.dwg
        dwi = self.dwi
        dwf = self.dwf
        dwo = self.dwo
        dwy = self.dwy

        dbg = self.dbg
        dbi = self.dbi
        dbf = self.dbf
        dbo = self.dbo
        dby = self.dby

        self.mwg += dwg * dwg
        self.mwi += dwi * dwi
        self.mwf += dwf * dwf
        self.mwo += dwo * dwo
        self.mwy += dwy * dwy
        self.wg += -self.lr * dwg / np.sqrt(self.mwg + 1e-8)
        self.wi += -self.lr * dwi / np.sqrt(self.mwi + 1e-8)
        self.wf += -self.lr * dwf / np.sqrt(self.mwf + 1e-8)
        self.wo += -self.lr * dwo / np.sqrt(self.mwo + 1e-8)
        self.wy += -self.lr * dwy / np.sqrt(self.mwy + 1e-8)

        self.mbg += dbg * dbg
        self.mbi += dbi * dbi
        self.mbf += dbf * dbf
        self.mbo += dbo * dbo
        self.mby += dby * dby
        self.bg += -self.lr * dbg / np.sqrt(self.mbg + 1e-8)
        self.bi += -self.lr * dbi / np.sqrt(self.mbi + 1e-8)
        self.bf += -self.lr * dbf / np.sqrt(self.mbf + 1e-8)
        self.bo += -self.lr * dbo / np.sqrt(self.mbo + 1e-8)
        self.by += -self.lr * dby / np.sqrt(self.mby + 1e-8)

        return smooth_loss

    def __init__(self, input_dim, output_dim, num_mem_cells=100, lr=0.1):
        ''' Initialize weights, bias,  character indexing
        @params:
        --------
            input_dim - length of the input sequence to the lstm
            num_mem_cells(optional)  - Number of memory cells, default num_mem_cells = 100
            lr(optional) - learning rate, default lr = 0.1
        '''

        self.lr = lr
        self.num_mem_cells = num_mem_cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat_dim = input_dim + num_mem_cells

        # Init parameters
        self.init_param(num_mem_cells, output_dim, self.concat_dim)


class LSTM_Node:
    ''' class LSTM_nodes holds the lstm state, parameters'''

    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input to node
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None
        self.y = None

    def forward_pass(self, x, s_prev=None, h_prev=None):
        ''' LSTM forward pass'''

        # At time t = 0
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        # save the state
        self.s_prev = s_prev
        self.h_prev = h_prev

        # input concantenation
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = np.tanh(self.state.s) * self.state.o
        # self.state.h = self.state.s * self.state.o
        self.state.y = np.dot(self.param.wy, self.state.h) + self.param.by
        pred = self.state.y
        self.state.prob = np.exp(pred) / np.sum(np.exp(pred))

        # store the inputs
        self.x = x
        self.xc = xc
        self.y = self.state.y


class LSTM_network:

    def __init__(self, input_dim, output_dim, seq_len, num_mem_cells=100, learning_rate=0.1):
        '''Initialize the LSTM unit, LSTM state'''

        # weights and bias are reused, so initialize lstm_param only once
        self.lstm_param = LSTM_param(input_dim, output_dim, num_mem_cells, learning_rate)
        # input sequence
        self.input_xs = []
        self.seq_len = seq_len
        # Node list
        self.lstm_node_list = []
        self.loss = 0
        self.num_mem_cells = num_mem_cells
        # loss
        self.smooth_loss = -np.log(1.0 / output_dim) * self.seq_len  # loss at iteration 0

    def apply_grad(self):
        # update your weights
        self.smooth_loss = self.lstm_param.apply_grad(self.loss, self.smooth_loss)

    def feed_backward(self, target):
        ''' Backpropagation '''

        assert len(self.lstm_node_list) == len(target)

        dh_next = np.zeros_like(self.lstm_node_list[0].state.h)
        ds_next = np.zeros_like(self.lstm_node_list[0].state.s)
        # weight gradient initialization
        idx = 0
        self.lstm_node_list[idx].param.dwg = np.zeros_like(self.lstm_node_list[idx].param.wg)
        self.lstm_node_list[idx].param.dwi = np.zeros_like(self.lstm_node_list[idx].param.wi)
        self.lstm_node_list[idx].param.dwf = np.zeros_like(self.lstm_node_list[idx].param.wf)
        self.lstm_node_list[idx].param.dwo = np.zeros_like(self.lstm_node_list[idx].param.wo)
        self.lstm_node_list[idx].param.dwy = np.zeros_like(self.lstm_node_list[idx].param.wy)

        # bias gradient initialization
        self.lstm_node_list[idx].param.dbg = np.zeros_like(self.lstm_node_list[idx].param.bg)
        self.lstm_node_list[idx].param.dbi = np.zeros_like(self.lstm_node_list[idx].param.bi)
        self.lstm_node_list[idx].param.dbf = np.zeros_like(self.lstm_node_list[idx].param.bf)
        self.lstm_node_list[idx].param.dbo = np.zeros_like(self.lstm_node_list[idx].param.bo)
        self.lstm_node_list[idx].param.dby = np.zeros_like(self.lstm_node_list[idx].param.by)

        for i, tt in reversed(list(enumerate(target))):
            # print('Param Address: {}, State Address: {}'.format(id(self.lstm_node_list[i].param), id(self.lstm_node_list[i].state)))
            param = self.lstm_node_list[i].param
            state = self.lstm_node_list[i].state
            x_dim = param.input_dim
            xc = self.lstm_node_list[i].xc

            dy = np.copy(self.lstm_node_list[i].state.prob)
            dy[tt] -= 1

            # gradient till cell state at time t
            dh = np.dot(param.wy.T, dy) + dh_next
            # ds = dh * state.o + ds_next
            ds = dh * state.o * (1 - (np.tanh(state.s) ** 2)) + ds_next

            # gradients till the non linearities for gates
            dg = state.i * ds
            di = state.g * ds
            df = self.lstm_node_list[i].s_prev * ds
            do = dh * state.s

            # gradients including non-linearities
            dg_input = (1.0 - state.g ** 2) * dg
            di_input = (1.0 - state.i) * state.i * di
            df_input = (1.0 - state.f) * state.f * df
            do_input = (1.0 - state.o) * state.o * do

            # Update gradients
            self.lstm_node_list[idx].param.dwy += np.outer(dy, state.h)
            self.lstm_node_list[idx].param.dby += dy
            self.lstm_node_list[idx].param.dwg += np.outer(dg_input, xc)
            self.lstm_node_list[idx].param.dwi += np.outer(di_input, xc)
            self.lstm_node_list[idx].param.dwf += np.outer(df_input, xc)
            self.lstm_node_list[idx].param.dwo += np.outer(do_input, xc)
            self.lstm_node_list[idx].param.dbg += dg_input
            self.lstm_node_list[idx].param.dbi += di_input
            self.lstm_node_list[idx].param.dbf += df_input
            self.lstm_node_list[idx].param.dbo += do_input

            ds_next = state.f * ds
            # compute bottom diff
            dxc = np.zeros_like(xc)
            dxc += np.dot(param.wi.T, di_input)
            dxc += np.dot(param.wf.T, df_input)
            dxc += np.dot(param.wo.T, do_input)
            dxc += np.dot(param.wg.T, dg_input)
            dh_next = dxc[x_dim:]

    def calculate_loss(self, target):
        ''' Cross entropy loss'''

        assert len(self.lstm_node_list) == len(target)

        loss = 0
        for i, tt in enumerate(target):
            prob = self.lstm_node_list[i].state.prob
            loss += -np.log(prob[tt])

        self.loss = loss

    def feed_forward(self, input_x, s_prev=None, h_prev=None):
        '''Storing input sequence, add new state everytime there is a new input
        @params:
        --------
            input_x - input vector to LSTM, x(t)
        '''

        self.input_xs.append(input_x)
        lstm_state = LSTM_state(self.lstm_param.input_dim, self.lstm_param.output_dim, self.lstm_param.num_mem_cells)
        self.lstm_node_list.append(LSTM_Node(self.lstm_param, lstm_state))

        idx = len(self.input_xs) - 1
        if idx == 0 and s_prev is None and h_prev is None:
            self.lstm_node_list[idx].forward_pass(input_x)
        elif idx == 0 and s_prev is not None and h_prev is not None:
            self.lstm_node_list[idx].forward_pass(input_x, s_prev, h_prev)
        else:
            self.s_prev = self.lstm_node_list[idx - 1].state.s
            self.h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].forward_pass(input_x, self.s_prev, self.h_prev)
            # self.s_prev = self.lstm_node_list[idx].state.s
            # self.h_prev = self.lstm_node_list[idx].state.h

    def reset(self):
        self.input_xs = []
        self.lstm_node_list = []


def gradient_check(inputs, targets, s_prev=None, h_prev=None):
    np.random.seed(initial_seed)
    input_dim = 65
    output_dim = 65
    num_samples = 25
    # input_dim, output_dim, num_iters, num_samples = objLstmData.vocab_size, objLstmData.vocab_size, 100, objLstmData.seq_len

    objLstmNet = LSTM_network(input_dim, output_dim, num_samples)
    objLstmNetOne = LSTM_network(input_dim, output_dim, num_samples)
    objLstmNetTwo = LSTM_network(input_dim, output_dim, num_samples)
    objLstmNetOne.lstm_param = objLstmNet.lstm_param
    objLstmNetTwo.lstm_param = objLstmNet.lstm_param

    x_list = inputs
    y_list = targets

    for i in range(num_samples):
        x_one_hot = np.zeros((output_dim))
        x_one_hot[x_list[i]] = 1
        objLstmNet.feed_forward(x_one_hot, s_prev, h_prev)

    objLstmNet.calculate_loss(y_list)
    objLstmNet.feed_backward(y_list)

    # Weights
    wg = objLstmNet.lstm_node_list[0].param.wg
    wi = objLstmNet.lstm_node_list[0].param.wi
    wf = objLstmNet.lstm_node_list[0].param.wf
    wo = objLstmNet.lstm_node_list[0].param.wo
    wy = objLstmNet.lstm_node_list[0].param.wy

    # Bias
    bg = objLstmNet.lstm_node_list[0].param.bg
    bi = objLstmNet.lstm_node_list[0].param.bi
    bf = objLstmNet.lstm_node_list[0].param.bf
    bo = objLstmNet.lstm_node_list[0].param.bo
    by = objLstmNet.lstm_node_list[0].param.by

    # Weights
    dwg = objLstmNet.lstm_node_list[0].param.dwg
    dwi = objLstmNet.lstm_node_list[0].param.dwi
    dwf = objLstmNet.lstm_node_list[0].param.dwf
    dwo = objLstmNet.lstm_node_list[0].param.dwo
    dwy = objLstmNet.lstm_node_list[0].param.dwy

    # Bias
    dbg = objLstmNet.lstm_node_list[0].param.dbg
    dbi = objLstmNet.lstm_node_list[0].param.dbi
    dbf = objLstmNet.lstm_node_list[0].param.dbf
    dbo = objLstmNet.lstm_node_list[0].param.dbo
    dby = objLstmNet.lstm_node_list[0].param.dby

    num_checks, delta, eps = 5, 1e-5, 1e-20
    for param, dparam, name in zip([wg, wi, wf, wo, wy, bg, bi, bf, bo, by], [dwg, dwi, dwf, dwo, dwy, dbg, dbi, dbf, dbo, dby], ['wg', 'wi', 'wf', 'wo', 'wy', 'bg', 'bi', 'bf', 'bo', 'by']):
        assert param.shape == dparam.shape, 'Error: Dimensions dont match'
        print name

        for i in xrange(num_checks):
            ri = int(uniform(0, param.size))
            old_val = param.flat[ri]

            # Feed one with +delta
            param.flat[ri] = old_val + delta
            for j in range(num_samples):
                x_one_hot = np.zeros((output_dim))
                x_one_hot[x_list[j]] = 1
                objLstmNetOne.feed_forward(x_one_hot, s_prev, h_prev)

            objLstmNetOne.calculate_loss(y_list)
            loss_one = objLstmNetOne.loss
            objLstmNetOne.reset()

            # Feed one with -delta
            param.flat[ri] = old_val - delta
            for j in range(num_samples):
                x_one_hot = np.zeros((output_dim))
                x_one_hot[x_list[j]] = 1
                objLstmNetTwo.feed_forward(x_one_hot, s_prev, h_prev)

            objLstmNetTwo.calculate_loss(y_list)
            loss_two = objLstmNetTwo.loss
            objLstmNetTwo.reset()
            param.flat[ri] = old_val

            grad_analytical = dparam.flat[ri]
            grad_numerical = (loss_one - loss_two) / (2.0 * delta)
            error = abs(grad_analytical - grad_numerical) / abs(grad_analytical + grad_numerical + eps)
            print '%f, %f => %e ' % (grad_numerical, grad_analytical, error)


def test():
    objLstm = LSTM('./input.txt')
    objLstm.train()


if __name__ == '__main__':
    test()
    # objLstmData = LSTM_grad_data('./input.txt')
    # x_list = objLstmData.inputs
    # y_list = objLstmData.targets
    # gradient_check(x_list, y_list, objLstmData)
