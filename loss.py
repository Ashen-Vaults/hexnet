
"""
A loss function measures how good our preditions are,
we can use this to adjust the parameters of out network
"""

import numpy as np
from hexnet.tenstor import Tensor


class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predictied: Tensor,
             actual: Tensor) -> Tensor: raise NotImplementedError


class MSE(Loss):
     """
     MSE is mean squared error,
     althought we're just going to do total squared error
      """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:

    	return np.sum((predicted - actual(** 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
			return 2 * (predicted - actual)
