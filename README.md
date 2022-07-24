# Adversarial Robust Deep Hedging

This repository solves problems of the type

Choose a hedging strategy $\phi$ such that it maximizes
$$
  \inf_{\mathbb{P}}R\Big(\sum_{t \in \mathcal{T}}\phi_t \Delta S^{\mathbb{P}}_t - C_T\Big) + \alpha(\mathbb{P})
$$
where 
