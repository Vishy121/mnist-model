import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mnist_model import MNIST_DNN, count_parameters
from utils.train_utils import train_model
import pytest
import torch

def test_parameter_count():
    model = MNIST_DNN()
    n_params = count_parameters(model)
    assert n_params < 25000, f"Model has {n_params} parameters (should be <25000)"

def test_input_output_shape():
    model = MNIST_DNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape {output.shape} is incorrect"

def test_model_accuracy():
    _, accuracy, _ = train_model(epochs=1)
    assert accuracy >= 85.0, f"Model accuracy {accuracy:.2f}% is below required 85%"

def test_model_forward():
    model = MNIST_DNN()
    batch_sizes = [1, 4, 16]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        y = model(x)
        assert y.shape == (batch_size, 10), f"Failed for batch size {batch_size}" 