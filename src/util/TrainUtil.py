from dataclasses import dataclass
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import List, Union, Optional, Any

from torch.utils.tensorboard import SummaryWriter

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
import torch

from src.dis.DeepHedge import DeepHedge


def save_trained_state(model: Any, directory: str, name: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory + '/' + name + '.pt')


def load_trained_state(model: Any, directory: str, name: str) -> None:
    model.load_state_dict(torch.load(directory + '/' + name + '.pt'))


class BaseVisualizer:

    def visualize(self, **kwargs) -> None:
        pass


@dataclass
class Visualizer(BaseVisualizer):

    freq: int = 200
    path_to_tensorboard: str = 'runs/deep_hedge' + datetime.now().strftime("%Y%m%d-%H%M%S")
    i: int = 0

    def __post_init__(self):
        self.writer = SummaryWriter(self.path_to_tensorboard)

    def visualize(self, **kwargs) -> None:
        self.i += 1

        loss_dict = kwargs['loss_dict']
        if not self.i % self.freq:
            return

        self.writer.add_scalars('Losses', loss_dict, self.i)


class NoLrSchedule:

    def step(self, **kwargs):
        pass


class ExponentialLrSchedule:

    def __init__(self, optimizer: Optimizer, gamma: float, step_sizes: Union[List[int], int]):
        self.lr_scheduler = ExponentialLR(optimizer, gamma)
        self.step_sizes = iter(step_sizes) if isinstance(step_sizes, list) else repeat(step_sizes)
        self.counter = 0
        self._next_step = next(self.step_sizes)
        self._last_step = 0

    def step(self, **kwargs):
        self.counter += 1
        if self.counter - self._last_step == self._next_step:
            self.lr_scheduler.step()
            self._last_step = self.counter
            self._next_step = next(self.step_sizes)


@dataclass
class LoadSaveConfig:
    save_to_directory: Optional[str]
    load_from_directory: Optional[str] = None
    save_every: Optional[int] = None

    def __post_init__(self):
        self.load_from_directory = self.load_from_directory if self.load_from_directory else self.save_to_directory
        self.save_every = self.save_every if self.save_to_directory else None


@dataclass
class DeepHedgeTrainer:
    deep_hedge: DeepHedge
    batch_sizes: List[int]
    lr_schedule: ExponentialLrSchedule
    ls_config: LoadSaveConfig
    visualizer: Visualizer = None
    tqdm: bool = True

    def __post_init__(self):
        self.visualizer = self.visualizer if self.visualizer else BaseVisualizer()
        self.losses = []

    def load_or_train(self):
        try:
            load_trained_state(self.deep_hedge.strategy, self.ls_config.load_from_directory, f'deep_hedge')
        except FileNotFoundError:
            self.train()

    def train(self):
        tqdm_batch_sizes = tqdm(self.batch_sizes) if self.tqdm else self.batch_sizes

        for i, batch_size in enumerate(tqdm_batch_sizes):
            loss = self.deep_hedge.step(batch_size).item()
            self.losses.append(loss)

            loss_dict = {'hedge_loss': loss}
            if self.tqdm:
                tqdm_batch_sizes.set_postfix(loss_dict)
            self.visualizer.visualize(i=i, loss_dict=loss_dict)

            if self.ls_config.save_every and i % (self.ls_config.save_every - 1) == 0:
                self.save_state()

        if self.ls_config.save_to_directory:
            self.save_state()

    def save_state(self):
        save_trained_state(self.deep_hedge.strategy, self.ls_config.save_to_directory, 'deep_hedge')


def train_trainers(ts, max_p):
    tasks = (train_trainer.remote(i=i, t=t) for i, t in enumerate(tqdm(ts)))
    return collect_trainings(tasks, max_p, len(ts))
