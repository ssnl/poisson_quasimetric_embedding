# Poisson Quasimetric Embedding (PQE)

**[Tongzhou Wang](https://ssnl.github.io), [Phillip Isola](http://web.mit.edu/phillipi/)**

---
:fire::fire: **Checkout [`torchqmet`](https://github.com/quasimetric-learning/torch-quasimetric): our new PyTorch package of many SOTA quasimetric learning methods, including [Interval Quasimetric Embeddings (IQEs)](https://tongzhouwang.info/interval_quasimetric_embedding), a new quasimetric learning method that improves performances with a much simpler form**!

---
This repository provides a PyTorch implementation of the Poisson Quasimetric Embedding (PQE) module for learning quasimetrics (i.e., asymmetrical distances).

PQEs are proposed in **On the Learning and Learnability of Quasimetrics (ICLR 2022)**. It is the first method that has known guarantee to well approximate any quasimetric, and can be trained in gradient-based optimization (e.g., deep learning). Other common choices (e.g., unconstrained deep networks) provably fail. Empirically, PQEs show good results on learning quasimetrics of various sizes and structures, including an large-scale social graph and an offline Q-learning task.

+ [arXiv](https://arxiv.org/abs/2206.15478)
+ [ICLR 2022 (OpenReview)](https://openreview.net/forum?id=y0VvIg25yk)
+ [Project Page](https://ssnl.github.io/quasimetric)

Requirements:
+ Python >= 3.7
+ PyTorch >= 1.9.0
+ NumPy >= 1.19.2

## Documentation

A PQE represents a quasimetric (i.e., distances that can be asymmetrical) over an embedding space.

The main entry point is the ``pqe.PQE`` class that initializes a latent PQE quasimetric.  This module takes in two generic latent vectors ``u`` and ``v`` (both of a specific size) and output the estimated quasimetric distance from ``u`` to ``v``.

Crucially, PQE can be trained to approximate any quasimetric distance. It does so by:
- Parametrizes several distributions of quasipartitions (a specific kinds of quasimetric)
    using Poisson Processes, where the latent vectors parametrize shapes (sets);
- For each quasipartition distribution, compute the expected value;
- Combine the expectations together together with non-negative weights (i.e., as a mixture
    of them);

```py
class PQE(nn.Module):
    def __init__(
        self,
        num_quasipartition_mixtures: int,                   # Number of mixtures of quasipartition distributions
                                                            #     to form the quasimetric estimate
        num_poisson_processes_per_quasipartition: int = 4,  # Number of Poisson processes used for each
                                                            #     quasipartition distributions
        measure: str = 'lebesgue',                          # Mean measure of the Poisson processes;
                                                            #     choices: 'lebesgue', 'gaussian'
        shape: str = 'halfline',                            # Parametrization of shapes (sets) in Poisson
                                                            #     process spaces by the input latents;
                                                            #     choices: 'halfline', 'gaussian'
        discounted: bool = False,                           # Whether this outputs distance or discounted
                                                            #     distance (no need to specify the base
                                                            #     because changing base is equivalent with
                                                            #     applying a scale, which is already learned)
    ):
        ...
```

### **Measures and shape parameterizations**

The default values `measure='lebesgue', shape='halfline'` means that it defaults to the PQE-LH variant (Eqn. (10)),
which is both straightforward to implement and works generally well in our experiments.

The choice `measure='lebesgue', shape='gaussian'` is prohibited as it leads to a very restricted symmetrical estimator.

Other choices will internally use a utility extension at [`pqe.cdf_ops`](./pqe/cdf_ops) (included in this repo, see [here](./pqe/cdf_ops) for details and documentations). Notably, `measure='gaussian', shape='gaussian'`  gives the PQE-GG variant used in paper.

## Getting Started and Examples

To start, we recommend adjusting `num_quasipartition_mixtures` according to the input latent
space dimensionality and keep the other arguments at default. The resulting PQE expects latent
vectors of size ``4 * num_quasipartition_mixtures``, and is a PQE-LH variant.

1. Quasimetric distances between two 64-dimensional latent vectors (with batch size 5)
    ```py
    >>> pqe = PQE(num_quasipartition_mixtures=16,
    ...           num_poisson_processes_per_quasipartition=4)
    >>> print(pqe)
    >>> u = torch.randn(5, 16 * 4)
    >>> v = torch.randn(5, 16 * 4)
    >>> print(pqe(u, v))  # non-negativity
    tensor([0.2902, 0.3297, 0.2735, 0.3561, 0.2403], grad_fn=<SqueezeBackward1>)
    >>> print(pqe(v, u))  # asymmetrical
    tensor([0.2554, 0.2326, 0.3263, 0.2562, 0.2975], grad_fn=<SqueezeBackward1>)
    >>> print(pqe(u, u))  # identity
    tensor([0., 0., 0., 0., 0.], grad_fn=<SqueezeBackward1>)
    >>> t = torch.randn(5, 16 * 4)
    >>> print(pqe(u, v) + pqe(v, t) >= pqe(u, t))  # triangle inequality
    tensor([True, True, True, True, True])
    ```

2. **Discounted** quasimetric distances between two 64-dimensional latent vectors (with batch size 5)
    ```py
    >>> discounted_pqe = PQE(num_quasipartition_mixtures=16,
    ...                      num_poisson_processes_per_quasipartition=4,
    ...                      discounted=True)
    >>> u = torch.randn(5, 16 * 4)
    >>> v = torch.randn(5, 16 * 4)
    >>> print(discounted_pqe(u, v))  # non-negativity (i.e., discounted distance <= 1)
    tensor([0.6986, 0.6614, 0.7698, 0.6864, 0.6138], grad_fn=<ProdBackward1>)
    >>> print(discounted_pqe(v, u))  # asymmetrical
    tensor([0.7258, 0.7232, 0.7233, 0.7440, 0.7511], grad_fn=<ProdBackward1>)
    >>> print(discounted_pqe(u, u))  # identity (i.e., discounted distance = 1)
    tensor([1., 1., 1., 1., 1.], grad_fn=<ProdBackward1>)
    >>> t = torch.randn(5, 16 * 4)
    >>> print(discounted_pqe(u, v) * discounted_pqe(v, t) <= discounted_pqe(u, t))  # triangle inequality
    tensor([True, True, True, True, True])
    ```

3. Components of a PQE
    ```py
    >>> pqe = PQE(num_quasipartition_mixtures=16,
    ...           num_poisson_processes_per_quasipartition=4)
    >>> print(pqe)
    PQE(
        num_quasipartition_mixtures=16
        num_poisson_processes_per_quasipartition=4
        num_effective_parameters=16
        discounted=False
        (measure): LebesgueMeasure()
        (shape): HalfLineShape()
        (quasipartition_aggregator): DistanceAggregator(
            (alpha_net): DeepLinearNet(
                bias=True, non_negative=True
                (mats): ParameterList(
                    (0): Parameter containing: [torch.FloatTensor of size 1x64]
                    (1): Parameter containing: [torch.FloatTensor of size 64x64]
                    (2): Parameter containing: [torch.FloatTensor of size 64x64]
                    (3): Parameter containing: [torch.FloatTensor of size 64x16]
                )
            )
        )
    )
    ```
    This PQE expects inputs from a (16*4)-dimensional latent space, where
     - each 4 consecutive dimensions are used for a quasipartition distribution (`num_poisson_process_per_quasipartition=4`)
     - a total of 16 such distributions mixed together (`num_quasipartition_mixtures=16`) to form the quasimetric
     - the mixture weights are given by `pqe.quasipartition_aggregator.alpha_net` which parametrizes the 16 non-negative
       numbers with deep linear networks (whose effective parameters are few).

4. Training a PQE
    ```py
    # Training with MSE on discounted distances

    # 128-dimensional paired data
    x = ...      # shape [N, 128]
    y = ...      # shape [N, 128]
    d_x2y = ...  # shape [N], >=0, could be +\infty

    # Simple 2 layer encoder from data space to latent space
    encoder = nn.Sequential(
        nn.Linear(128, 512),
        nn.ReLU()
        nn.Linear(512, 512),
        nn.ReLU()
        nn.Linear(512, 16 * 4),
    )

    # Define the latent PQE, use discounted version
    discounted_pqe = PQE(num_quasipartition_mixtures=16,
                         num_poisson_processes_per_quasipartition=4,
                         discounted=True)
    optim = torch.optim.Adam(
        itertools.chain(encoder.parameters(), discounted_pqe.parameters()),  # both encoder and pqe have parameters
        lr=1e-3,
    )
    while keep_training():
        optim.zero_grad()
        zx = encoder(x)
        zy = encoder(y)
        pred = discounted_pqe(zx, zy)
        loss = F.mse_loss(0.9 ** d_x2y, pred)
        loss.backward()
        optim.step()
    ```

## Citation

Tongzhou Wang, Phillip Isola. "On the Learning and Learnability of Quasimetrics" International Conference on Learning Representations. 2022.

```
@inproceedings{wang2022learning,
  title={On the Learning and Learnability of Quasimetrics},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## Questions

For questions about the code provided in this repository, please open an GitHub issue.

For questions about the paper, please contact Tongzhou Wang (`tongzhou _AT_ mit _DOT_ edu`).
