import torch
import torch.nn as nn

from .quasipartition_aggregators import QuasipartitionAggregatorBase, DistanceAggregator, DiscountedDistanceAggregator
from .measures import MeasureBase, LebesgueMeasure, GaussianBasedMeasure
from .shapes import ShapeBase, HalfLineShape, GaussianShape
from .utils import get_num_effective_parameters


class PQE(nn.Module):
    r"""
    Poisson Quasimetric Embedding (PQE).


    A PQE represents a quasimetric (i.e., distances that can be asymmetrical) over an embedding
    space.  This module thus takes in two generic latent vectors ``u`` and ``v`` (both of a specific
    size) and output the estimated quasimetric distance from ``u`` to ``v``.

    Crucially, PQE can be trained to approximate any quasimetric distance. It does so by:
        - Parametrizes several distributions of quasipartitions (a specific kinds of quasimetric)
          using Poisson Processes, where the latent vectors parametrize shapes (sets);
        - For each quasipartition distribution, compute the expected value;
        - Combine the expectations together together with non-negative weights (i.e., as a mixture
          of them);

    To start, we recommend adjusting :attr:`num_quasipartition_mixtures` according to the input latent
    space dimensionality and keep the other arguments at default. The resulting PQE expects latent
    vectors of size ``4 * num_quasipartition_mixtures``, and is a PQE-LH variant (Eqn. (10)), which
    generally works well in our experiments.

    For details on these parametrizations, see Sec. 4 of our paper.

    Args:
        num_quasipartition_mixtures (int):  Number of quasipartition distributions whose expected
                                            value are mixted together to obtain the distance
                                            estimate.
        num_poisson_processes_per_quasipartition (int):  Number of Poisson processes used to
                                                         parametrize a quasipartition distribution.
                                                         Default: 4.
        measure (str):  Measure used in the Poisson processes. Choices are ``"lebesgue"`` and
                        ``"guassian"``.  Default: ``"lebesgue"``.
        shape (str):  Shape parametrizations used in the Poisson processes. Choices are ``"halfline"`` and
                      ``"guassian"``.  ``"guassian"`` can only be used with ``"guassian"`` measure.
                      Default: ``"halfline"``.
        discounted (bool):  If set, this module instead estimates discounted distances (with the
                            base in ``(0, 1)``). This is particularly useful when training data has
                            many large distances (especially with infinities), when larger
                            distances are less important to be approximated accurately, or when the
                            approximate target is inherently discounted (such as Q functions).
                            Default: False.

    Shape:
        - Input: Two tensors of shape ``(..., num_quasipartition_mixtures * num_poisson_processes_per_quasipartition)``
        - Output: ``(...)``

    Attributes:
        num_quasipartition_mixtures (int)
        num_poisson_processes_per_quasipartition (int)
        discounted (bool)
        measure (nn.Module): A module that represents the Poisson processes measure choice.
        shape (nn.Module): A module that represents the shape parametrization from the input latents to the Poisson
                           process spaces.

    Examples::

        >>> pqe = PQE(num_quasipartition_mixtures=16, num_poisson_processes_per_quasipartition=4)
        >>> u = torch.randn(5, 16 * 4)
        >>> v = torch.randn(5, 16 * 4)
        >>> print(pqe(u, v))
        tensor([0.2902, 0.3297, 0.2735, 0.3561, 0.2403], grad_fn=<SqueezeBackward1>)
        >>> print(pqe(v, u))
        tensor([0.2554, 0.2326, 0.3263, 0.2562, 0.2975], grad_fn=<SqueezeBackward1>)
        >>> print(pqe(u, u))
        tensor([0., 0., 0., 0., 0.], grad_fn=<SqueezeBackward1>)

        >>> # Discounted
        >>> discounted_pqe = PQE(num_quasipartition_mixtures=16, num_poisson_processes_per_quasipartition=4, discounted=True)
        >>> u = torch.randn(5, 16 * 4)
        >>> v = torch.randn(5, 16 * 4)
        >>> print(discounted_pqe(u, v))
        tensor([0.6986, 0.6614, 0.7698, 0.6864, 0.6138], grad_fn=<ProdBackward1>)
        >>> print(discounted_pqe(v, u))
        tensor([0.7258, 0.7232, 0.7233, 0.7440, 0.7511], grad_fn=<ProdBackward1>)
        >>> print(discounted_pqe(u, u))
        tensor([1., 1., 1., 1., 1.], grad_fn=<ProdBackward1>)
    """

    quasipartition_aggregator: QuasipartitionAggregatorBase
    measure: MeasureBase
    shape: ShapeBase

    def __init__(self,
                 num_quasipartition_mixtures: int,
                 num_poisson_processes_per_quasipartition: int = 4,
                 measure: str = 'lebesgue',
                 shape: str = 'halfline',
                 discounted: bool = False):
        super().__init__()
        assert num_quasipartition_mixtures > 0, "num_quasipartition_mixtures must be positive"
        assert num_poisson_processes_per_quasipartition > 0, "num_poisson_processes_per_quasipartition must be positive"
        self.latent_dim = num_quasipartition_mixtures * num_poisson_processes_per_quasipartition
        # Will need to reshape the latents to be 2D so that
        #   - the last dim represents Poisson processes that parametrize a distribution of quasipartitions
        #   - the second last dim represents the number of mixtures of such quasipartition distributions
        self.latent_2d_shape = torch.Size([num_quasipartition_mixtures, num_poisson_processes_per_quasipartition])
        if measure == 'lebesgue':
            self.measure = LebesgueMeasure(shape=self.latent_2d_shape)
        elif measure == 'gaussian':
            self.measure = GaussianBasedMeasure(shape=self.latent_2d_shape)
        else:
            raise ValueError(f'Unsupported measure={measure}')
        if shape == 'halfline':
            self.shape = HalfLineShape()
        elif shape == 'gaussian':
            self.shape = GaussianShape()
        else:
            raise ValueError(f'Unsupported shape={shape}')
        self.discounted = discounted
        if discounted:
            self.quasipartition_aggregator = DiscountedDistanceAggregator(num_quasipartition_mixtures=num_quasipartition_mixtures)
        else:
            self.quasipartition_aggregator = DistanceAggregator(num_quasipartition_mixtures=num_quasipartition_mixtures)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert u.shape[-1] == v.shape[-1] == self.latent_dim
        expected_quasipartitions = self.shape.expected_quasipartiton(
            u.unflatten(-1, self.latent_2d_shape),
            v.unflatten(-1, self.latent_2d_shape),
            measure=self.measure,
        )
        return self.quasipartition_aggregator(expected_quasipartitions)

    def num_effective_parameters(self):
        return get_num_effective_parameters(self)

    def extra_repr(self) -> str:
        return '\n'.join([
            f'num_quasipartition_mixtures={self.latent_2d_shape[0]}',
            f'num_poisson_processes_per_quasipartition={self.latent_2d_shape[1]}',
            f'num_effective_parameters={self.num_effective_parameters()}',
            f'discounted={self.discounted}',
        ])

__all__ = ['PQE']
