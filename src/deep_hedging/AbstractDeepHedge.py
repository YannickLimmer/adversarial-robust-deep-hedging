from _py_abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic

import torch
from torch import nn

from src.deep_hedging.StrategyNet import StrategyNetConfig, StrategyNet
from src.derivative.Derivative import Derivative


@dataclass
class DeepHedgeConfig:

    """
    Fundamental deep hedge configuration. A dataclass that contains all necessary information for a general deep hedge
    application.

    :param derivative: A derivative which will be replicated in the to be configured deep hege.
    :type derivative: Derivative
    :param initial_asset_price: The initial asset price of the asset price process.
    :type initial_asset_price: torch.Tensor
    :param strategy_config: Configuration of the strategy net.
    :type strategy_config: StrategyNetConfig
    """

    derivative: Derivative
    initial_asset_price: torch.Tensor
    strategy_config: StrategyNetConfig

    def __post_init__(self):
        if self.initial_asset_price.shape[0] != self.strategy_config.dimension_of_asset:
            raise AttributeError(
                'Initial asset price dimension does not coincide with asset price dimension of strategy specs.',
            )


_DeepHedgeConfig = TypeVar('_DeepHedgeConfig', bound=DeepHedgeConfig)


class AbstractDeepHedge(nn.Module, Generic[_DeepHedgeConfig], metaclass=ABCMeta):

    """
    A base class for deep hedging, that implements the deep hedging procedure proposed by Buehler et al in an abstract
    manner.

    :param config: A generic configuration class. Provides all required information for deep hedging.
    :type config: Generic[_DeepHedgeConfig]
    """

    def __init__(self, config: _DeepHedgeConfig):
        super().__init__()
        self.config = config
        self.td = config.derivative.td

        self.strategy_layer = StrategyNet(self.config.strategy_config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Implements the deep hedge logic. During one forward pass, three elements are tracked through time: the
        information, the strategies based on that information, and the resulting wealth originating from that strategy.
        If the module is in training mode, i.e. the `training` attribute evaluates as `True`, a simple profit and loss
        is returned. If the strategy is in evaluation mode, the strategy is returned.

        The base implementation operates as follows. Initially, the initial information and wealth process are
        constructed. This step also ensures that the initial information contains the time of the current time step.
        Then, moving forward in time, the information is fed into the `strategy_layer`, what results into the strategy
        for the corresponding time step. This is used to update the wealth process. Eventually, the information is
        updated for the next time step, what includes parsing the corresponding time. After all time steps are handled,
        the payoffs are weighed against the derivative payoffs, and a profit and loss value is computed.

        :param inputs: A tensor that contains enough data to provide the information for the strategy layer at each time
        step and to construct the wealth process of a chosen strategy. In the base implementation, this tensor is of
        shape (n, m, d',), where n is the batch size, m the number of time steps, and d' some integer greater or equal
        to the dimension of the asset. The latter allows to parse information that is not tradable to the strategy.

        For the respective structure of the input tensor, the methods
            - `_extract_initial_information`
            - `_create_initial_wealth`
            - `_update_information`
            - `_update_wealth`
            - `_calculate_profit_and_loss`
            - `_terminal_asset_price_from_information`
        have to be adapted correspondingly when subclassing.
        :type inputs: torch.Tensor
        :return: If `training` is `True` the profit and loss of the prevailing strategy, which is a Tensor of shape (n,)
        where n is the batch size. Otherwise, the strategy for the scenarios of the batch are returned, which is a
        tensor of shape (n, m, d,), where n is the batch size, m the number of time steps, and d the dimension of the
        asset.
        :rtype: torch.Tensor
        """
        information = self._extract_initial_information(inputs)
        wealth = self._create_initial_wealth(inputs)

        strategy = []
        for time_step_index in self.td.indices:
            strategy.append(self.strategy_layer(self._filter_information(information)))
            information = self._update_information(information, inputs, time_step_index)
            wealth = self._update_wealth(wealth, inputs, strategy[-1], time_step_index)

        profit_and_loss = self._calculate_profit_and_loss(information, wealth)

        if self.training:
            return self._prepare_pnl_for_training(profit_and_loss)

        return torch.stack(strategy, dim=1)

    @abstractmethod
    def _extract_initial_information(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Given the inputs, the initial information is extracted, which is (at least partially, see filtering) passed to
        the `strategy_layer` in the first iteration. Encodes the time of the first time step.

        :param inputs: The input tensor passed to the deep hedge.
        :type inputs: torch.Tensor
        :return: The initial information that contains the time of the first time step. A tensor of shape (n, d',),
        where n is the batch size and d' the dimension of the information process, which is usually greater than the
        dimension of the asset. It is advised that one entry of the second axis is used to parse the initial time.
        :rtype: torch.Tensor
        """
        pass

    # noinspection PyMethodMayBeStatic
    def _create_initial_wealth(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Creates the initial values of the value process.

        :param inputs: The input tensor passed to the deep hedge.
        :type inputs: torch.Tensor
        :return: The initial wealth. A tensor of shape (n,) where n is the batch-size. The base implementation returns a
        zero-valued tensor of this shape, however, implementations that respect the derivative price should include it
        here.
        :rtype: torch.Tensor
        """
        return torch.zeros((inputs.shape[0]))

    @abstractmethod
    def _update_information(
            self,
            information: torch.Tensor,
            inputs: torch.Tensor,
            time_step_index: int,
    ) -> torch.Tensor:
        """
        Yields a tensor containing the information for the next time step.

        :param information: The (unfiltered) information of the previous time step, a tensor of shape (n, d',), where n
        is the batch size and d' the dimension of the information process.
        :type information: torch.Tensor
        :param inputs: The inputs passed to the deep hedge.
        :type inputs: torch.Tensor
        :param time_step_index: The time step index of the current time period. Gives full information of time when used
        on the `td` attribute.
        :type time_step_index: int
        :return: The (unfiltered) information for the next time step, a tensor of shape (n, d',), where n is the batch
        size and d' the dimension of the information process.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def _filter_information(self, information: torch.Tensor) -> torch.Tensor:
        """
        Before passed to the `strategy_layer` the information of the prevailing time step may be reduced or altered,
        allowing Platonic market structures and, particularly, non-tradable information.
        :param information: The current state of the information process. A tensor of shape (n, d',), where n is the
        batch size and d' the dimension of the information process.
        :type information: torch.Tensor
        :return: The filtered information, a tensor of shape (n, d'',), where d'' is the dimension of the filtered
        information process.
        :rtype: torch.Tensor
        """
        pass

    @staticmethod
    @abstractmethod
    def _update_wealth(
            wealth: torch.Tensor,
            inputs: torch.Tensor,
            strategy_for_time_step: torch.Tensor,
            time_step_index: int,
    ) -> torch.Tensor:
        """
        Updates the wealth process according to the strategy of the prevailing time step.

        :param wealth: The previous wealth value, a tensor of shape (n,), with n being the batch size.
        :type wealth: torch.Tensor
        :param inputs: The inputs passed to the deep hedge.
        :type inputs: torch.Tensor
        :param strategy_for_time_step: The strategy that resulted for this time step. A tensor of shape (n, d,), where n
        is the batch size and d the dimension of tradable assets.
        :type strategy_for_time_step: torch.Tensor
        :param time_step_index: The time step index of the current time period. Gives full information of time when used
        on the `td` attribute.
        :type time_step_index: int
        :return: The updated wealth value a tensor of shape (n,), with n being the batch size.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def _calculate_profit_and_loss(self, information: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
        """
        Computes the profit and loss from the wealth and information.

        :param information: The values of the information process. A tensor of shape (n, d',), where n is the batch size
        and d' the dimension of the information process.
        :type information: torch.Tensor
        :param wealth: The wealth, a tensor of shape (n,), where n is the batch size.
        :type wealth: torch.Tensor
        :return: The profit and loss, being the wealth minus the derivative payoff. A tensor of shape (n,), where n is
        the batch size.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def _terminal_asset_price_from_information(self, information: torch.Tensor) -> torch.Tensor:
        """
        Reduces the terminal asset price contained in the information process to the tradable components.
        """
        pass

    @abstractmethod
    def _prepare_pnl_for_training(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        """
        Allows to transform the profit and loss values prior to outputting.
        """
        pass
