import numpy as np

class Conv3x3:
    ''' A convolution layers using 3x3 filters.
    '''

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # Filters is a 3D array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        ''' Generates all possible 3x3 image regions using valid padding.
            - `image` is a 2D numpy array
        ''' 
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[i:(i + 3), j:(j + 3)]
                yield img_region, i, j

    def forward(self, input):
        ''' Performs a forward pass of the conv layer using `input`.
            Returns a 3D numpy array with dims (h, w, num_filters).
            - `input` is a 2D numpy array
        '''
        self.last_input = input
        
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(img_region * self.filters, axis=(1,2))

        return output

    def backprop(self, dL_dout, learning_rate):
        ''' Performs a backward pass of the conv layer.
            - `dL_dout` is the loss gradient for this layer's outputs.
            - `learning_rate` is a float.
        '''
        dL_dfilters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i, j, f] * img_region

        # Update filters
        self.filters -= learning_rate * dL_dfilters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None