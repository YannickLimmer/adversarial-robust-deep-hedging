from abc import ABCMeta, abstractmethod

from src.util.torch_util.TrainingUtil import Metrics, Trainer


class Callback(metaclass=ABCMeta):

    def __call__(self, metrics: Metrics, trainer: Trainer) -> None:
        return self._call(metrics, trainer)

    @abstractmethod
    def _call(self, metrics: Metrics, trainer: Trainer) -> None:
        pass


class PrintMetrics(Callback):

    def _call(self, metrics: Metrics, trainer: Trainer) -> None:
        metrics = [f'{k}: {v: .6f}'for k, v in metrics.create_print_dict().items()]
        print(f"Iter: {trainer.counter}  \t{'    '.join(metrics)}")


class PrintGeneratorParameters(Callback):

    def _call(self, metrics: Metrics, trainer: Trainer) -> None:
        n_params = [f"{n.split('.')[-1]}: {p: .6f}" for n, p in trainer.gen_config.generator.named_parameters()]
        print(f"Iter: {trainer.counter}   \t{'    '.join(n_params)}")


class PrintEmptyLine(Callback):

    def _call(self, metrics: Metrics, trainer: Trainer) -> None:
        print()
