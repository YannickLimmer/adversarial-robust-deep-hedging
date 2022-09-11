import torch

from src.deep_hedging.AbstractDeepHedge import DeepHedgeConfig, AbstractDeepHedge


class DeepHedge(AbstractDeepHedge[DeepHedgeConfig]):

    def _extract_initial_information(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Constructs a tensor in the shape of the information process containing the initial time and the initial asset
        price value for each scenario.

        :param inputs: The input tensor passed to the deep hedge. A tensor of shape (n, m, d' - 1,), where n is the
        batch size, m the number of time steps, and d' the dimension of the information process (containing a time
        component).
        :type inputs: torch.Tensor
        :return: The initial information that contains the time of the first time step. A tensor of shape (n, d',),
        where n is the batch size and d' the dimension of the information process. The time component is added at
        position 0 of axis 1.
        :rtype: torch.Tensor
        """
        return torch.cat(
            (
                self.td.times[0] * torch.ones_like(inputs[:, 0, 0:1]),
                self.config.initial_information_value * torch.ones_like(inputs[:, 0, :]),
            ),
            dim=1,
        )

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
        :param inputs: The increments of the information process without time. A tensor of shape (n, m, d' - 1,), where
        n is the batch size, m the number of time steps, and d' the dimension of the information process (containing a
        time component).
        :type inputs: torch.Tensor
        :param time_step_index: The time step index of the current time period. Gives full information of time when used
        on the `td` attribute.
        :type time_step_index: int
        :return: The (unfiltered) information for the next time step, a tensor of shape (n, d',), where n is the batch
        size and d' the dimension of the information process. This is obtained by adding the increments of the
        information process and updating the time in the first entry along the second axis.
        :rtype: torch.Tensor
        """
        return torch.cat(
            (
                self.td.times[time_step_index + 1] * torch.ones_like(inputs[:, 0, 0:1]),
                information[:, 1:] + inputs[:, time_step_index],
            ),
            dim=1,
        )

    def _filter_information(self, information: torch.Tensor) -> torch.Tensor:
        """
        The identity, no filtering takes place.
        """
        return information

    @staticmethod
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
        :param inputs: The increments of the asset price process. A tensor of shape (n, m, d,), where n is the batch
        size, m the number of time steps, and d the dimension of the tradable asset.
        :type inputs: torch.Tensor
        :param strategy_for_time_step: The strategy that resulted for this time step. A tensor of shape (n, d,), where n
        is the batch size and d the dimension of tradable assets.
        :type strategy_for_time_step: torch.Tensor
        :param time_step_index: The time step index of the current time period. Gives full information of time when used
        on the `td` attribute.
        :type time_step_index: int
        :return: The updated wealth value a tensor of shape (n,), with n being the batch size. The value is computed by
        adding the summed, and by strategy scaled, increments to the wealth of the previous period.
        :rtype: torch.Tensor
        """
        return wealth + torch.sum(strategy_for_time_step * inputs[:, time_step_index], dim=1)

    def _calculate_profit_and_loss(self, information: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
        derivative_payoff = self.config.derivative.payoff_for_terminal_asset_values(
            self._terminal_asset_price_from_information(information),
        )
        return self.config.derivative.price + wealth - derivative_payoff

    def _terminal_asset_price_from_information(self, information: torch.Tensor) -> torch.Tensor:
        return information[:, 1:]

    def _prepare_pnl_for_training(self, profit_and_loss: torch.Tensor) -> torch.Tensor:
        return profit_and_loss
