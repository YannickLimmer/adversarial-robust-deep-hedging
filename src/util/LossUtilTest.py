import torch

from src.util.LossUtil import conditional_value_at_risk, negative_standard_deviation


def test_std():
    arr = torch.arange(100, dtype=torch.float).reshape((10, 10))
    assert negative_standard_deviation(negative_standard_deviation(arr, dim=1), dim=0).item() == -2.5131524239441205e-07


def test_cvar():
    arr = torch.arange(100, dtype=torch.float32)
    arr = arr.reshape((10, 10))
    assert torch.equal(
        input=conditional_value_at_risk(arr, .0, dim=1),
        other=torch.arange(start=0, end=100, step=10, dtype=torch.float32),
    )
    assert torch.equal(
        input=conditional_value_at_risk(arr, .0, dim=0),
        other=torch.arange(start=0, end=10, step=1, dtype=torch.float32),
    )
    assert torch.equal(
        input=conditional_value_at_risk(arr, .2, dim=1),
        other=torch.arange(start=1, end=101, step=10, dtype=torch.float32),
    )
    assert torch.equal(
        input=conditional_value_at_risk(arr, .29, dim=1),
        other=torch.arange(start=1, end=101, step=10, dtype=torch.float32),
    )
