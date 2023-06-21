---
title: '“Hindsight” – An easy yet effective RL Technique HER with Pytorch implementation'
date: 2022-05-18
permalink: /posts/2022/05/hindsight/
tags:
  - RL
  - HER
  - Pytorch
---

This week, I will share a paper published by OpenAI at NeurIPS 2017. The ideas presented in this paper are quite insightful, and it tackles a complex problem using only simple algorithmic improvements. I gained significant inspiration from this paper. At the end, I will also provide a brief implementation of HER (Hindsight Experience Replay).

> Original Paper Information <br> **Title**: Hindsight experience replay <br> **Author**: Marcin Andrychowicz | Filip Wolski | Alex Ray | Jonas Schneider | Rachel Fong | Peter Welinder | Bob McGrew | Josh Tobin | OpenAI Pieter Abbeel | Wojciech Zaremba <br> **Code**: https://github.com/openai/baselines/blob/master/baselines/her/README.md

# Background
The combination of reinforcement learning and neural networks has achieved success in various domains involving sequential decision-making, such as Atari games, Go, and robot control tasks.

Typically, in reinforcement learning tasks, a crucial aspect is designing a reward function that reflects the task itself and guides the optimization of the policy. Designing a reward function can be highly complex, which limits the application of reinforcement learning in the real world. It requires not only understanding of the algorithm itself but also substantial domain-specific knowledge. Moreover, in scenarios where it is difficult for us to determine what actions are appropriate, it is challenging to design an appropriate reward function. Therefore, algorithms that can learn policies from rewards that do not require explicit design, such as binary variables indicating task completion, are important for applications.

One capability that humans possess but most model-free reinforcement learning algorithms lack is the ability to learn from "hindsight". For example, if a basketball shot misses to the right, a reinforcement learning algorithm would conclude that the sequence of actions associated with the shot is unlikely to lead to success. However, an alternative "hindsight" conclusion can be drawn, namely that if the basket were slightly to the right (to the location where the ball landed), the same sequence of actions would result in success.

The paper introduces a technique called Hindsight Experience Replay (HER), which can be combined with any off-policy RL algorithm. HER not only improves sampling efficiency but, more importantly, enables the algorithm to learn policies from binary and sparse rewards. HER incorporates the current state and a target state as inputs during replay, where the core idea is to replay an episode (a complete trajectory) using a different goal than the one the agent was originally trying to achieve.

Check my post on Discovery Lab: [【每周一读】“事后诸葛亮”——一种简单有效的强化学习技术HER（文末附Pytorch代码）](https://mp.weixin.qq.com/s/CCDmxhc79WTWAnImsegvhQ)