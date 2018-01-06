"""
Our neural nets will made up of layers.
Each layers needs to pass its inputs forward
and propagate graddients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""

import numpy as np
from hexnet.tensor import Tensor


class Layer:
    
    
    def __init__(self) -> None:
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagete this gradient through the layer
        """
        raise NotImplementedError

