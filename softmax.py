import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        ''' Performs a forward pass of the softmax layer using the given input.
            Returns a 1d numpy array containing the respective probability values.
            - `input` can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape # cache input.shape before flattening

        input = input.flatten()
        self.last_input = input # cache input after flattening

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals # cache totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, dL_dout, lr):
        ''' Performs a backward pass of the softmax layer.
            Returns the loss gradient for this layer's inputs.
            - `dL_dout` is the loss gradient for this layer's outputs.
            - `lr` is a float
        '''
        # We know only 1 element of dL_dout will be nonzero
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            dout_dt = -t_exp[i] * t_exp / (S ** 2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights
            # Gradients of loss against totals
            dL_dt = grad * dout_dt
            # Gradients of loss against weights/biases/input
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            # Update parameters
            self.weights -= lr * dL_dw
            self.biases -= lr * dL_db
            return dL_dinputs.reshape(self.last_input_shape)