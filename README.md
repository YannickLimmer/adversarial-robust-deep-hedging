# Adversarial Robust Deep Hedging

This repository solves problems of the type

Choose a hedging strategy $\phi$ such that it maximizes

$$
\inf_{P}R\Big(\sum_{t \in T}\phi_t \Delta S^{P}_t - C_T\Big) + \alpha(P)
$$

where 
