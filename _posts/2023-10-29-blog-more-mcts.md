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

# EfficientZero
The author finds that in limited-data setting, for instance, Atari 100K, which is 2 hours of real-time game experience, MuZero is not that impressive in performance. This leads the author to observe 3 issues. 

1. Lack of supervision on environment model. In previous MCTS RL algorithms, the environment model is either given or only trained with rewards, values, and policies, which cannot provide sufficient training signals due to their scalar nature. The problem is more severe when the reward is sparse or the bootstrapped value is not accurate. The MCTS policy improvement operator heavily relies on the environment model. Thus, it is vital to have an accurate one.
2. Predicting the reward from a state is a hard problem. If we only see the first observation, along with future actions, it is very hard both for an agent and a human to predict at which exact future timestep the player would lose a point. However, it is easy to predict the agent will miss the ball after a sufficient number of timesteps if he does not move. In practice, a human will never try to predict the exact step that he loses the point but will imagine over a longer horizon and thus get a more confident prediction. 
3. This value target suffers from off-policy issues, since the trajectory is rolled out using an older policy, and thus the value target is no longer accurate. When data is limited, we have to reuse the data sampled from a much older policy, thus exaggerating the inaccurate value target issue.

<img src="/images/effi-limi2.png" alt="bit-flipping" width="500" height="300" style="margin-left: auto; margin-right: auto; display: block;">
<figure>
  <figcaption style='text-align: center'>Figure 1. To predict at which exact future timestep the player would lose a point could be hard, but it is easy to predict the agent will miss the ball after a sufficient number of timesteps if he does not move </figcaption>
</figure>

Then 3 methods are proposed to solve each of the issues. The first one is self-supervised to provide more information. The idea is illustrated by the right-side figure. That is, the hidden state from dynamic function should be similar to the hidden state from the representation of the next observation. 
The second is to predict value prefix, which is discounted sum of rewards, instead of stepwise reward. 
The third is to re-run MCTS for an adjusted TD steps based on the age of the trajectory.
