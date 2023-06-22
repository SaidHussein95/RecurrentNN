import numpy as np
from Base import BaseLayer

class TanH(BaseLayer):

    def __int__(self):
        super().__init__()
        self.f_tensor = 0
    def forward(self, input_tensor):
        self.f_tensor = np.tanh(input_tensor)
        return self.f_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor*(1 - np.sqare(self.f_tensor))
        return error_tensor