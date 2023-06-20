---
title: "Temporal Graph Attention Network Prediction on Ethereum Transaction Cost and Analysis on 'The Merge'"
excerpt: "Proposed a GNN model based on temporal transaction network to predict Ethereum Transaction Cost<br/><img src='/images/model_struct.png' width='500' height='300'>"
collection: portfolio
---

The Merge aimed to resolve the disadvantages of ETH 1.0 of low scalability and high energy consumption. By merging the PoS Beacon Chain to the main chain, the Merge reduced about 99.95\% Ethereum’s energy consumption.
Our goal is to analyze this big event for Ethereum. We carry out descriptive analysis comparing data before and after the Merge. As for application, we present a GNN model based on the embedding of transaction networks. We are interested in the Gas Price, which is of great economic value, and ‘The Merge’ takes one step closer to sharding, which will increase the processing speed of the Ethereum mainnet and lower the transaction fees. 

Based on prior works, we intend to combine temporal and graphical information to improve prediction performance. We apply graph attention convolution (GATConv) and global pooling to obtain embeddings of the entire graph, and causal Transformer to learn temporal dependencies.

# Data
We used [Ethereum ETL](https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/) to manipulate and integrate our data into our database in Google Cloud Platform. There are about 1.1 million transactions generated each day, and we collected about 250 GB over the past few months. 

## Data Preprocessing
<figure>
  <img
  src="/images/dis_trans_b.png"
  alt="number of transactions per block">
  <img
  src="/images/dis_trans_t.png"
  alt="number of transactions every 2 minutes">
  <figcaption style='text-align: center'> Distribution of number of transactions in each block vs Distribution of number of transactions in a time window of 2 minutes</figcaption>
</figure>

The first step is to aggregate transactions by time. The number of transactions per block can vary dramatically, ranging from 1 transaction to approximately 900 transactions in one single block. The vastly changing size of blocks presents challenges to accurately predicting the minimal gas price in each block. However, the number of transactions against time shows consistency. In practice, it is more important to consider the delay rather than the actual block. Therefore, we aggregate the transactions by a non-overlap sliding window with a length of 2 minutes.

<figure>
  <img
  src="/images/des_pre.png"
  alt="Description of feature scales. The scales vary vastly.">
  <figcaption style='text-align: center'> Description of feature scales. The scales vary vastly.</figcaption>
</figure>

The second step is data normalization. The raw data we collected has features on different scales. The gas price and value are both in GWei, and since 1 GWei equals 0.00000000119 ETH, both features contain great values. The value transferred on average is 6.76E+17 and the mean of gas price is 1.78E+10, while the average of gas is 2.11E+5. Data normalization can be crucial to the model's performance. We used Z-score normalization for each feature separately.

The last step is to mini-batch along the diagonal for consecutive graphs to create a giant graph. The data loader is built on these giant graphs to preserve the order of sequence.

# Method
## Descriptive Analysis: Transaction Graphs Comparison

## Predictive Analysis: Gas Price Prediction
### Label
to label the data. For each block, only transactions processed were recorded, so that the minimum gas price in each block is the lowest gas price for the transaction to be considered(in this block). After aggregation and normalization, we compute the minimum gas price $m_i$ within each graph $i$. Then we compute $y_i=\min_{k\in\{i+1,i+2,...,i+l\} )}m_k $, which is the minimum of consecutive minimum gas prices.
# Result
## Analysis on "The Merge"
## Gas Price Prediction