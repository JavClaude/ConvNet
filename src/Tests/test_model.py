import torch
from src.Model.baseline import ConvNet

device = "cpu"

def test_forward_pass():
    test_model = ConvNet(1, 9)
    test_inputs = torch.randn(size=(64, 1, 28, 28), device=device, dtype=torch.float)
    model_outputs = test_model(test_inputs)
    assert len(model_outputs) == 2
