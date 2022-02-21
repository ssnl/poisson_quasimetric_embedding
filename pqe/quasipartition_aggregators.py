import torch
import torch.nn as nn
import numpy as np

from .utils import DeepLinearNet, sigmoid_pow, multidot


class QuasipartitionAggregatorBase(nn.Module):
    def __init__(self, num_quasipartition_mixtures: int):
        super().__init__()
        self.num_quasipartition_mixtures = num_quasipartition_mixtures


class DistanceAggregator(QuasipartitionAggregatorBase):
    def __init__(self, num_quasipartition_mixtures: int):
        super().__init__(num_quasipartition_mixtures)
        self.alpha_net = DeepLinearNet(input_dim=num_quasipartition_mixtures, output_dim=1, non_negative=True)

    def forward(self, expected_quasipartitions: torch.Tensor) -> torch.Tensor:
        return self.alpha_net(expected_quasipartitions).squeeze(-1)


class DiscountedDistanceAggregator(QuasipartitionAggregatorBase):
    # Sec. C.4.2
    def __init__(self, num_quasipartition_mixtures: int):
        super().__init__(num_quasipartition_mixtures)
        self.beta_net = DeepLinearNet(input_dim=1, output_dim=num_quasipartition_mixtures, non_negative=False)

        # Initialize logits so initial output is between 0.5 and 0.75. (Sec. C.4.3)
        #
        # Note that this is important since we are multiplying a bunch of things < 1 together,
        # and thus must take care to not make the result close to 0.
        #
        # Say the quasipartitions are 0.5. For output = y, with k quasipartitions,
        # we want the base to be roughly
        #   - log ( y^{-2/k} - 1).

        k = num_quasipartition_mixtures
        low_out = 0.5
        high_out = 0.75
        low = -np.log(low_out ** (-2 / k) - 1)
        high = -np.log(high_out ** (-2 / k) - 1)
        # `DeepLinearNet` should initialize s.t. the collapsed vector roughly
        # has zero mean and 1 variance. This holds even for intermediate activations.
        # NB that we crucially used `in_dim=1` rather than `out_dim=1`, which will make
        # weights have variance O(1/n).

        ms = list(self.beta_net.mats)

        with torch.no_grad():
            # collapse all but last
            out_before_last: torch.Tensor = multidot(ms[1:])
            norm_out_before_last = out_before_last.norm()
            unit_out_before_last = out_before_last/ out_before_last.norm()

            # now simply constrain the projection dimension
            ms[0].sub_((ms[0] @ unit_out_before_last) @ unit_out_before_last.T) \
                    .add_(torch.empty(k, 1).uniform_(low, high).div(norm_out_before_last) @ unit_out_before_last.T)  # noqa: E501
            q = self.beta_net.collapse().squeeze(1).sigmoid().pow(0.5).prod().item()
            assert low_out <= q <= high_out, q

    def forward(self, expected_quasipartitions: torch.Tensor) -> torch.Tensor:
        logits = self.beta_net.collapse().squeeze(1)
        return sigmoid_pow(logits, expected_quasipartitions).prod(-1)
