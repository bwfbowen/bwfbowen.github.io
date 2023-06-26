---
title: 'Will DRL Make Profit in High-Frequency Trading?'
date: 2021-10-21
permalink: /posts/2021/10/drlhft/
tags:
  - RL
  - HFT
  - LOB
---

Can deep reinforcement learning algorithms be used to train a trading agent that can achieve long-term profitability using Limit Order Book (LOB) data? To answer this question, this article proposes a deep reinforcement learning framework for high-frequency trading and conducts experiments using limit order data from [LOBSTER](https://lobsterdata.com) with the PPO algorithm. The results show that the agent is able to identify short-term patterns in the data and propose profitable trading strategies.

> Original Paper Information <br> **Title**: Deep Reinforcement Learning for Active High Frequency Trading <br> **Author**: Antonio Briola, Jeremy Turiel, Riccardo Marcaccioli, Tomaso Aste <br> **DOI**: arXiv:2101.07107

# Background
## Limit Order Book

A Limit Order Book (LOB) records the outstanding limit orders in the current market. A limit order is an order with a specified price that is not lower than the specified price. The LOB data used in the article refers to the ten-level market data, which includes the top ten buy and sell prices with the highest priority, as well as the total order quantity at each price level. The difference between the best bid price and the best ask price is referred to as the bid-ask spread.

<figure>
  <img
  src="/images/lob-drlhft.png"
  alt="A demonstration of LOB data">
  <figcaption style='text-align: center'>Figure 1. A demonstration of LOB data </figcaption>
</figure>