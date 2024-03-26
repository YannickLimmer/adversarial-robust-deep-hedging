import torch
import torch.nn as nn


class TransformedParameter(nn.Module):
    def __init__(self, data: torch.Tensor, requires_grad=True, input_transform=None, output_transform=None):
        super(TransformedParameter, self).__init__()
        self.raw = nn.Parameter(data if input_transform is None else input_transform(data), requires_grad=requires_grad)
        self.input_transform = input_transform
        self.output_transform = output_transform

    @property
    def val(self):
        if self.output_transform is not None:
            return self.output_transform(self.raw)
        return self.raw

    @val.setter
    def val(self, value):
        if self.input_transform is not None:
            self.raw.data = self.input_transform(value)
        else:
            self.raw.data = value


class ExpParameter(TransformedParameter):

    def __init__(self, data: torch.Tensor, requires_grad=True,):
        super().__init__(data, requires_grad, torch.log, torch.exp)


class TanhParameter(TransformedParameter):

    def __init__(self, data: torch.Tensor, requires_grad=True,):
        super().__init__(data, requires_grad, torch.arctanh, torch.tanh)
