import numpy as np

class Optimizer:
    def __int__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):

    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is None:
            new_weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            new_weight_tensor = shrink - (self.learning_rate * gradient_tensor)

        return new_weight_tensor


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.momentum_term = (self.momentum_rate * self.momentum_term) - (self.learning_rate * gradient_tensor)
        if self.regularizer is None:
            weight_tensor = weight_tensor + self.momentum_term
        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            weight_tensor = shrink + self.momentum_term
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, lr, mu, rho):
        super().__init__()
        self.learning_rate = lr
        self.mu = mu
        self.rho = rho
        self.v_term = 0
        self.r_term = 0
        self.it = 1
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.v_term = (self.mu * self.v_term) + ((1 - self.mu) * gradient_tensor)
        self.r_term = self.rho * self.r_term + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        v_hat = self.v_term / (1 - math.pow(self.mu, self.it))
        r_hat = self.r_term / (1 - math.pow(self.rho, self.it))

        adam_term = v_hat / (np.sqrt(r_hat) + self.epsilon)

        if self.regularizer is None:
            weight_tensor = weight_tensor - self.learning_rate * adam_term

        else:
            shrink = weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            weight_tensor = shrink - self.learning_rate * adam_term

        self.it = self.it + 1

        return weight_tensor