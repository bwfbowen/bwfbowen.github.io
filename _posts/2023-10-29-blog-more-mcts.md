---
title: 'Some recent advancement around MuZero'
date: 2023-10-29
permalink: /posts/2023/10/more-mcts/
tags:
  - EfficientZero
  - SampledMuZero
  - StochasticMuZero
  - MCTS
---

<img src="/images/limit-muzero.png" alt="Limitation of MuZero" width='300' height='200'>

I have found 4 interesting papers that discover and address the limitations of MuZero. [EfficientZero](https://proceedings.neurips.cc/paper_files/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html) and [MCTS as regularized policy optimization](https://proceedings.mlr.press/v119/grill20a.html) improve the algorithm itself; [SampledMuZero](https://proceedings.mlr.press/v139/hubert21a.html) and [StochasticMuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) extend the algorithm into a new setting that the original algorithm is not able to perform well. 

# EfficientZero[Link](https://proceedings.neurips.cc/paper_files/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html)
The author finds that in limited-data setting, for instance, Atari 100K, which is 2 hours of real-time game experience, MuZero is not that impressive in performance. This leads the author to observe 3 issues. 

1. Lack of supervision on environment model. In previous MCTS RL algorithms, the environment model is either given or only trained with rewards, values, and policies, which cannot provide sufficient training signals due to their scalar nature. The problem is more severe when the reward is sparse or the bootstrapped value is not accurate. The MCTS policy improvement operator heavily relies on the environment model. Thus, it is vital to have an accurate one.
2. Predicting the reward from a state is a hard problem. If we only see the first observation, along with future actions, it is very hard both for an agent and a human to predict at which exact future timestep the player would lose a point. However, it is easy to predict the agent will miss the ball after a sufficient number of timesteps if he does not move. In practice, a human will never try to predict the exact step that he loses the point but will imagine over a longer horizon and thus get a more confident prediction. 
3. This value target suffers from off-policy issues, since the trajectory is rolled out using an older policy, and thus the value target is no longer accurate. When data is limited, we have to reuse the data sampled from a much older policy, thus exaggerating the inaccurate value target issue.

<img src="/images/effi-limi2.png" alt="bit-flipping" width="500" height="300" style="margin-left: auto; margin-right: auto; display: block;">
<figure>
  <figcaption style='text-align: center'>Figure 1. To predict at which exact future timestep the player would lose a point could be hard, but it is easy to predict the agent will miss the ball after a sufficient number of timesteps if he does not move </figcaption>
</figure>

Then 3 methods are proposed to solve each of the issues. 

The first one is self-supervised to provide more information. The idea is illustrated by the Figure 2. That is, the hidden state from dynamic function should be similar to the hidden state from the representation of the next observation. 

<img src="/images/self-consistent.png" alt="bit-flipping" width="500" height="300" style="margin-left: auto; margin-right: auto; display: block;">
<figure>
  <figcaption style='text-align: center'>Figure 2. The self-supervised consistency loss </figcaption>
</figure>

The second is to predict value prefix, which is discounted sum of rewards, instead of stepwise reward. The author chose the LSTM for value prefix prediction. During the training time, the LSTM is supervised at every time step, since the value prefix can be computed whenever a new state comes in. This per-step rich supervision allows the LSTM can be trained well even with limited data. 

The third is to re-run MCTS for an adjusted TD steps based on the age of the trajectory. the value target is computed by sampling a trajectory from the replay buffer and computing: $$z_t=\sum_{i=0}^{k-1}\gamma^i u_{t+i}+\gamma^k v_{t+k}$$ This value target suffers from off-policy issues, since the trajectory is rolled out using an older policy, and thus the value target is no longer accurate. The author proposed to use rewards of a dynamic horizon $l$ from the old trajectory, where $l<k$ and $l$ should be smaller if the trajectory is older. 

# MCTS as regularized policy optimization[Link](https://proceedings.mlr.press/v119/grill20a.html)

This paper focuses on policy improvement, and proposed a new method to *act*, *search* and *update* for AlphaZero. 

They started from policy optimization, which aims at finding a globally optimal policy $\pi_\theta$, generally using iterative updates. Each updates the current policy $\pi_\theta$ by solving a local maximization problem of the form $$\pi_{\theta'}≜\arg\max_{\textbf{y}\in \mathcal{S}}\mathcal{Q}^T_{\pi_\theta}\textbf{y}-\mathcal{R}(\textbf{y}, \pi_\theta)$$, several model-free algorithms employ such policy optimization, for instance, TRPO and MPO set $\mathcal{R}$ to be the KL-divergence between consecutive policies. With idea from policy optimization, they proposed a new policy for AlphaZero: $$\newcommand{\kl}{\mathrm{KL}} \bar{\pi}\triangleq\arg\max_{\textbf{y}\in \mathcal{S}}[\textbf{q}^T\textbf{y}-\lambda_N\kl[\pi_\theta,\textbf{y}] ] $$, which in practical can be computed as: $$\bar{\pi}=λ_N\frac{\pi_θ}{α-\textbf{q}}$$, where $\alpha$ is used to make $\bar{\pi}$ a proper probability vector and is found through dichotomic search over $(\alpha_{min},\alpha_{max})$, whose value is defined as: $$\alpha_{min}≜\max_{b\in \mathcal{A}}(q[b]+\lambda_N·\pi_\theta[b]), α_{max}≜\max_{b\in\mathcal{A}}q[b]+\lambda_N$$.

Then they argued that the empirical visit count distribution $\hat{\pi}$: $$\hat{\pi}(a\mid x)≜\frac{1+n(x,a)}{\mid \mathcal{A}\mid \sum_bn(x,b)}$$, has several shortcomings. The first is the efficiency. When a promising new (high-value) leaf is discovered, many additional simulations might be needed before this information is reflected in $\hat{\pi}$; since $\bar{\pi}$ is directly computed from Q-values, this information is updated instantly. The second is the expressiveness. $\hat{\pi}$ is simply the ratio of two integers, and thus has limited expressiveness when the simulation budget is low. The third is that the prior $\pi_\theta$ is trained against the target $\hat{\pi}$, but the latter is only improved for actions that have been sampled at least once during search. This may be problematic for certain actions that would require a large simulation budget to be sampled even once. 

The new policy is applied for acting, searching and learning process of AlphaZero.
1. ACT: AlphaZero acts in the real environment by sampling actions according to $$a ∼ \hat{π}(·\ x_{root})$$. Instead, they proposed to sample actions sampling according to $$a ∼ \bar{π}(·\mid x_{root})$$. 
2. SEARCH: During search, they proposed to stochastically sample actions according to $\bar{\pi}$ instead of the deterministic action selection rule. At each node x in the tree, $\bar{π}(·)$ is computed with Q-values and total visit counts at the node based on Definition 1. We label this variant as SEARCH. 
3. LEARN: AlphaZero computes locally improved policy with tree search and distills such improved policy into $π_θ$. They proposed to use $\bar{π}$ as the target policy in place of $\hat{π}$ to train prior policy 

# Stochastic MuZero[Link](https://openreview.net/forum?id=X6D9bAHhBQ1)

# Sampled MuZero[Link](https://proceedings.mlr.press/v139/hubert21a.html)