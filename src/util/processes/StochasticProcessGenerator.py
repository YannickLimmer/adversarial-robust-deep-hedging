from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from numpy.typing import NDArray


class StochasticProcessGenerator(metaclass=ABCMeta):

    def generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
            random_number_generator: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """
        Generates samples of the stochastic process of the shape (n, m+1, d,), where n is the number of initial values
        (also referred to as number of realizations),  m+1 the number of times (and hence m the number of time steps),
        and d the dimension of the asset.

        :param initial_value: An array of shape (n,d,). Describes the initial values of the process. To obtain multiple
            paths per initial value, repeat the respective value in this array and reconstruct it from the first axis of
            the output.
        :type initial_value: NDArray
        :param times: An array of shape (m,). The first entry corresponds to the initial time and the last one to the
            terminal time.
        :type times: NDArray
        :param stochastic_increments: Optional array of a shape that is individual to the process class. Contains the
            random numbers that are used to generate the process. This is a feature that allows the user to use a random
            number generator other than numpy's default or reuse random numbers.
        :type stochastic_increments: Optional[NDArray]
        :param random_number_generator: The RNG used for generation of the stochastic increments. Only used if
            stochastic increments are not specified. Defaults to a default generator with default seed.
        :type random_number_generator: Optional[np.random.Generator]
        :return: An array of the shape (n, m+1, d,). Realizations of the stochastic process.
        :rtype: NDArray
        """
        number_of_realizations = initial_value.shape[0]
        number_of_dimensions = initial_value.shape[1]
        number_of_time_steps = times.shape[0]-1
        if stochastic_increments is not None:
            self.verify_dimensionality_of_stochastic_increment(
                number_of_dimensions,
                number_of_realizations,
                number_of_time_steps,
                stochastic_increments,
            )
        else:
            stochastic_increments = self.generate_stochastic_increments(
                number_of_realizations,
                number_of_time_steps,
                number_of_dimensions,
                random_number_generator,
            )
        return self._generate(initial_value, times, stochastic_increments)

    @abstractmethod
    def _generate(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
    ) -> NDArray:
        pass

    @abstractmethod
    def shape_of_stochastic_increments(
            self,
            number_of_realizations: int,
            number_of_time_steps: int,
            number_of_dimensions: int,
    ) -> Tuple:
        pass

    def generate_stochastic_increments(
            self,
            number_of_realizations: int,
            number_of_time_steps: int,
            number_of_dimensions: int,
            rng: Optional[np.random.Generator],
    ) -> NDArray:
        rng = rng if rng is not None else np.random.default_rng()
        return self._generate_stochastic_increments(
            number_of_realizations,
            number_of_time_steps,
            number_of_dimensions,
            rng,
        )

    @abstractmethod
    def _generate_stochastic_increments(
            self,
            number_of_realizations: int,
            number_of_time_steps: int,
            number_of_dimensions: int,
            rng: np.random.Generator,
    ) -> NDArray:
        pass

    def verify_dimensionality_of_stochastic_increment(
            self,
            number_of_dimensions,
            number_of_realizations,
            number_of_time_steps,
            stochastic_increments
    ):
        target_shape = self.shape_of_stochastic_increments(
            number_of_realizations,
            number_of_time_steps,
            number_of_dimensions,
        )
        if stochastic_increments.shape != target_shape:
            raise AttributeError(
                f'Stochastic increments were provided with shape {stochastic_increments.shape}, '
                f'but {target_shape} was expected.'
            )

    def provide_generator(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
            random_number_generator: Optional[np.random.Generator] = None,
    ) -> Callable[[int], torch.Tensor]:

        def generator(n: int) -> torch.Tensor:
            return torch.as_tensor(self.generate(
                np.ones((n, 1)) * initial_value[None, :],
                times,
                stochastic_increments,
                random_number_generator
            ), dtype=torch.float32)

        return generator

    def provide_increment_generator(
            self,
            initial_value: NDArray,
            times: NDArray,
            stochastic_increments: Optional[NDArray] = None,
            random_number_generator: Optional[np.random.Generator] = None,
    ) -> Callable[[int], torch.Tensor]:

        def generator(n: int) -> torch.Tensor:
            return torch.diff(torch.as_tensor(self.generate(
                np.ones((n, 1)) * initial_value[None, :],
                times,
                stochastic_increments,
                random_number_generator
            ), dtype=torch.float32), 1, 1)

        return generator
