from Base import BaseLayer
from FullyConnected import FullyConnected
from TanH import TanH
from Sigmoid import Sigmoid
import numpy as np


class RNN(BaseLayer):
     def __init__(self, input_size, hidden_size, output_size):
         self.trainable = True
         self.regular_loss = None

         self.input_size = input_size
         self.hidden_size = hidden_size
         self.output_size = output_size
         self.hidden_state = np.zeros(hidden_size)

         self._memorize = False
         self._optimizer = None
         self._gradient_weights = None

         self.tanh = TanH()
         self.sigmoid = Sigmoid()

         self.ful_con_layer_hidden = FullyConnected(self.hidden_size +self.input_size, self.hidden_size)
         self.ful_con_layer_output = FullyConnected(self.hidden_size,self.output_size)



     def forward(self, input_tensor):
         self.sigmoid_activations = []
         self.tanh_activations = []
         self.output_ful_con_input_tensors = []
         self.ful_con_layer_hiddens_input_tensors = []
         self.ful_con_layer_hidden_gradient_weights = []
         if self.memorize:
             last_hidden_layer = self.hidden_state
         else:
             last_hidden_layer = np.zeros(self.hidden_size)

         batch = input_tensor.shape[0]
         output_tensor = np.ndarray((batch, self.output_size))

         for b in range(batch):
             x_t = np.concatenate([last_hidden_layer,input_tensor[b]]).reshape(1, -1)
             tanh_tensor = self.ful_con_layer_hidden.forward(x_t)
             current_hidden_state = self.tanh.forward(tanh_tensor)

             #update last hidden layer
             last_hidden_layer = current_hidden_state[0]

             # transition of hy
             sigmoid_tensor =self.ful_con_layer_output.forward(current_hidden_state)
             sigmoid_output_tensor = self.sigmoid.forward(sigmoid_tensor)

             output_tensor[b] = sigmoid_output_tensor[0]

             # append

             self.ful_con_layer_hiddens_input_tensors.append(self.ful_con_layer_hidden.input_tensor)

             self.output_ful_con_input_tensors.append(self.ful_con_layer_output.input_tensor)
             self.sigmoid_activations.append(self.sigmoid.activations)
             self.tanh_activations.append(self.tanh.activations)

         #update hissen state
             self.hidden_state = current_hidden_state[0]

         return output_tensor

     def backward(self, error_tensor):
         self.gradient_weights =np.zeros_like(self.ful_con_layer_hidden.weights)
         self.ful_con_layer_output_gradient_weights =np.zeros_like(self.ful_con_layer_output.weights)
         grad_last_hid_layer = 0
         batch = error_tensor.shape[0]
         gradient_inputs = np.zeros((batch, self.input_size))

         time_step = batch - 1

         while time_step >= 0:

             #output
             self.sigmoid.activations =self.sigmoid_activations[time_step]
             self.ful_con_layer_output.input_tensor =self.output_ful_con_input_tensors[time_step]
             #use sigmoid error as input
             ful_con_layer_output_error =self.ful_con_layer_output.backward(self.sigmoid.backward(error_tensor[time_step]))

             #hidden
             self.tanh.activations = self.tanh_activations[time_step]
             self.ful_con_layer_hidden.input_tensor =self.ful_con_layer_hiddens_input_tensors[time_step]
             # use tanh error as input
             ful_con_layer_hidden_error =self.ful_con_layer_hidden.backward(self.tanh.backward(ful_con_layer_output_error + grad_last_hid_layer))

             #gradient last hiden layer
             grad_last_hid_layer = ful_con_layer_hidden_error[:,:self.hidden_size]

             #gradient with respect to input
             gradient_inputs[time_step] = ful_con_layer_hidden_error[:,self.hidden_size:][0]

             #update
             self.gradient_weights +=self.ful_con_layer_hidden.gradient_weights
             self.ful_con_layer_output_gradient_weights +=self.ful_con_layer_output.gradient_weights

             #update time step
             time_step -= 1

         if self.optimizer:
             self.ful_con_layer_output.weights =self.optimizer.calculate_update(self.ful_con_layer_output.weights, self.ful_con_layer_output_gradient_weights)
             self.weights = self.optimizer.calculate_update(self.weights,self.gradient_weights) #hidden

         return gradient_inputs

     def calculate_regularization_loss(self):
         if self.optimizer.regularizer:
             self.regular_loss += self.optimizer.regularizer.norm( self.ful_con_layer_hidden.weights) +self.optimizer.regularizer.norm(self.ful_con_layer_output.weights)
         return self.regular_loss

     def initialize(self, weights_initializer, bias_initializer):
         self.ful_con_layer_hidden.initialize(weights_initializer,bias_initializer)
         self.ful_con_layer_output.initialize(weights_initializer,bias_initializer)

     def _memorize_getter(self):
         return self._memorize

     def _memorize_setter(self, value):
         self._memorize = value

     memorize = property(
         fget=_memorize_getter,
         fset=_memorize_setter
     )

     def _optimizer_getter(self):
         return self._optimizer

     def _optimizer_setter(self, value):
         self._optimizer = value

     optimizer = property(
         fget=_optimizer_getter,
         fset=_optimizer_setter
     )

     def _weight_getter(self):
         return self.ful_con_layer_hidden.weights

     def _weight_setter(self, value):
         self.ful_con_layer_hidden.weights = value

     weights = property(
         fget=_weight_getter,
         fset=_weight_setter
     )

     def _gradient_weights_getter(self):
         return self._gradient_weights

     def _gradient_weights_setter(self, value):
         self._gradient_weights = value

     gradient_weights = property(
         fget=_gradient_weights_getter,
         fset=_gradient_weights_setter
     )
