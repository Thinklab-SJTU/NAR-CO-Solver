# One-Shot-CardNN-Solver
This is the official implementation of our ICLR 2023 paper "Towards One-shot Neural Combinatorial Solvers: Theoretical and Empirical Notes on the Cardinality-Constrained Case". 

* [[paper]](https://openreview.net/pdf?id=h21yJhdzbwz)

This work is jointly done by [ThinkLab@SJTU](http://thinklab.sjtu.edu.cn) and [JD Explore Academy](https://corporate.jd.com/).

This repository offers neural network solvers, datasets, training and evaluation protocols
for three cardinality-constrained combinatorial optimization problems: 
* Facility Location Problem (FLP), on synthetic data and real-world Starbucks locations;
* Max Covering Problem (MCP), on synthetic data and real-world Twitch datasets;
* Portfolio Optimization (PortOpt), on real-world stock prices in 2021.

## A Brief Introduction of our Paper

As read from the title, our ultimate goal is building _one-shot neural combinatorial solvers_:
a neural network whose input is the parameters of a combinatorial optimization (CO) problem, 
and whose output is the solution (i.e. the decision variables). The neural network is expected 
to output the solution in one-shot, instead of a tedious multi-step auto-regressive manner. 
We believe such one-shot neural solvers have the following advantages over traditional solvers:
* Higher efficiency (neural solvers on GPU vs traditional solvers on CPU)
* Enabling joint predict-and-optimize paradigms (differentiable neural solvers vs usually 
  non-differentiable traditional solvers)

Towards the ultimate goal, we identify the following technical challenges: the output of neural 
networks is usually unconstrained, yet the solutions to combinatorial optimization usually have
complicated constraints. Besides, the discrete nature of combinatorial constraints conflicts 
with the continuous nature of neural networks. To resolve these issues, this paper champions 
the following methodology: softly enforcing constraints to neural networks by a differentiable
layer. The overview of such a pipeline is shown as follows.

![overview](imgs/one-shot-nn.png)

The key is developing the differentiable constraint layer and designing efficient 
self-supervised loss (usually by estimating the objective score). Such a flexible design also
enables gradient-based update at testing time.

Seeing that a general method to handle all CO problems seems too challenging, this paper 
focuses on developing a more practical paradigm for solving the cardinality-constrained CO. We 
present a differentiable layer named CardNN to handle cardinality constraints, which is based 
on Sinkhorn iterations and Gumbel trick, and conduct theoretical study of the design of the 
constraint-enforcing layer. We are not diving too deep into the theoretical results of
Gumbel-Sinkhorn in this introduction, and here summarizes our main conclusion: 

> A tighter **constraint violation** leads to better performance for one-shot neural 
> combinatorial solvers.

We believe such theoretical insights could further generalize to other CO problems beyond 
cardinality constraints.

In this paper, we compare three ways of handling cardinality constraints, theoretically:

* **Erdos Goes Neural** [(Karalias & Loukas, NeurIPS'20)](https://proceedings.neurips.cc/paper_files/paper/2020/file/49f85a9ed090b20c8bed85a5923c669f-Paper.pdf)
  puts constraint violation as a penalty term in the loss. Theoretically, the neural net's output
  can be arbitrary thus the constraint violation is unbonded.
* **SOFT-TopK** [(Xie et al., NeurIPS'20)](https://proceedings.neurips.cc/paper_files/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)
  uses Sinkhorn to enforce cardinality constraint, which is equivalent to a TopK selection. 
  SOFT-TopK offers an upper-bound of constraint violation, yet the bound can diverge in the wost
  case (when k-th and (k+1)-th elements are equal).
* **Gumbel-Sinkhorn-TopK** (this paper) further addresses the diverging issue in SOFT-TopK. 
  Specifically, the constraint violation can be arbitrarily controlled, and its theoretical 
  upper bound is tighter than SOFT-TopK by introducing the Gumbel trick.
  
Our experiment shows consistent result with the theoretical study: empirically, we have Erdos
Goes Neural (EGN) < SOFT-TopK (CardNN-S) < Gumbel-Sinkhorn-TopK (CardNN-GS). Experiments are 
conducted on both synthetic problems (in line with most neural CO solver papers), and transfer
learning from synthetic problems to real-world instances. 

## Experiment on Pure CO Problems

Pure CO problem means the problem parameters are known at the decision-making time. The 
neural solver should learn to optimize the objective score.

In experiments, considering both efficiency and objective score, our neural solvers can be 
_comparative to state-of-the-art traditional solvers (Gurobi and SCIP), and sometimes can even
surpass_.

### Facility Location Problem (FLP)

Given m locations, we want to extend k facilities such that the goods can be stored at the
nearest facility and delivered more efficiently. Another scenario may be a city with m 
communities, and we want to build k facilities (like battery swapping stations for e-vehicles)
for all residents.

The objective of FLP is to minimize the sum of the distances between each location and its 
nearest facility. See the paper for the math form.

### Max Covering Problem (MCP)

Given m sets and n objects, each set may cover any number of objects and each object is
associated with a value. MCP aims to find k sets (k ≪ m) such that the covered objects have 
the maximum sum of values. This problem reflects real-world scenarios such as discovering 
influential seed users in social networks.

See the paper for the math form.

### Results

Since there lacks large-scale real-world benchmark for FLP and MCP, we follow most neural CO
papers and test our solvers on synthetic data. The synthetic results are shown as follows. 
Our neural solvers are the blue ones.

![synthetic results](imgs/synthetic-results.png)

We also consider a transfer learning setting with real-world data: neural networks are firstly 
learned with synthetic data, and then tested on real-world datasets. The real-world datasets 
are collected from the [Starbucks locations](https://www.kaggle.com/datasets/kukuroo3/starbucks-locations-worldwide-2021-version)
in London, New York, Shanghai, Seoul with 166-569 stores, and the 
[social network collected from Twitch](https://snap.stanford.edu/data/twitch-social-networks.html).
Our neural solvers surpasses Gurobi & SCIP on FLP. On MCP, our neural solvers are inferior, 
while it is worth noting that our Homotopy version **CardNN-HGS** finds all optimal solutions,
which is hard to achieve for most previous neural solvers.

![real world results](imgs/real-world-results.png)


## Experiment on Predictive CO Problems

Predictive CO problem means the problem parameters are unknown at the decision-making time.
We consider the portfolio optimization problem as an example: when making an investment, we 
cannot know the returns/risks of assets in the future. The only feasible way is predicting 
the future returns/risks, while such prediction always contains errors. If the optimizer is 
unaware of such prediction error, even the optimal solution does not reflect the optimal 
decision in the future, and it is very likely that the optimizer is misled by the prediction 
error.

The above challenge give rise to the recent research topic of joint "predict and optimize", 
and here we show that our neural solver also owns the flexibility to handle predictive CO by 
jointly learning a neural predictor and a neural solver end-to-end.

### Cardinality-constrained Portfolio Optimization

Cardinality constrained portfolio optimization considers a practical scenario where a 
portfolio must contain no more than k assets. Restricting the number of assets in a portfolio 
can reduce operational costs. A good portfolio should have a high return (measured by mean 
vector µ) and low risk (measured by covariance matrix Σ).

In this paper, we refer to maximizing the Nobel-prize-winning Sharpe ratio (see the 
paper for the math form). An LSTM is built as the predictor, followed by fully-connected 
layers as the neural solver.

### Results

We test on the real prices of S&P 500 assets in 2021. The joint predict-and-optimize method 
based on our CardNN-GS outperforms others with higher returns and lower risks.

![portfolio optimization results](imgs/port-results.png)

Visualization also shows that the portfolio built by CardNN-GS is closer to the efficient 
frontier. And note that reaching the efficient frontier is nearly impossible in practice
because we do not know the real returns/risks at the decision-making time.

![portfolio optimization visualization](imgs/port-visual.png)

## Get Started

To run the code, first install ``torch`` and ``torch_geometric`` according to the official 
docs ([[PyTorch]](https://pytorch.org/get-started/locally/), [[Geometric]](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).
If you face any issues, there are many online resources available.

Then install the following Python packages:

```
pip install ortools matplotlib yfinance pandas cvxpylayers xlwt
```

## Run the Experiments

Follow the next steps and you shall reproduce the main results in our paper.

### Facility Location Problem (FLP)

On synthetic data, 500 locations.
```bash
python facility_location_experiment.py --cfg cfg/facility_location_rand500.yaml
```

On synthetic data, 800 locations.
```bash
python facility_location_experiment.py --cfg cfg/facility_location_rand800.yaml
```

On real-world data (Starbucks), distance measured by Euclidean distance. 
```bash
python facility_location_experiment.py --cfg cfg/facility_location_starbucks_euclidean.yaml
```

On real-world data (Starbucks), distance measured by Manhattan distance.
```bash
python facility_location_experiment.py --cfg cfg/facility_location_starbucks_manhattan.yaml
```

### Max Covering Problem (MCP)

On synthetic data, 500 sets and 1000 items.
```bash
python max_covering_experiment.py --cfg cfg/max_cover_rand500.yaml
```

On synthetic data, 1000 sets and 2000 items.
```bash
python max_covering_experiment.py --cfg cfg/max_cover_rand1000.yaml
```

On real-world data (Twitch social network).
```bash
python max_covering_experiment.py --cfg cfg/max_cover_twitch.yaml
```

### Portfolio Optimization

Note that portfolio optimization experiment does not offer a configuration system. Some hyperparameters 
are coded at the beginning of ``portfolio_opt_experiment.py``.

Training:

```bash
python portfolio_opt_experiment.py --train --method predict-and-opt
```
``--method`` can be selected from ``predict-and-opt`` (LSTM+our CardNN-GS), ``predict-then-opt``
(LSTM+Gurobi), ``history-opt``. You need to install ``gurobipy`` and have a valid license to run 
``predict-then-opt``.

After training, load a checkpoint (at 55-th epoch here) and run testing:

```bash
python portfolio_opt_experiment.py --start_epoch 55 --method predict-and-opt
```

## Citation
If you find our paper/code useful in your research, please cite

```
@inproceedings{wang2023cardinality,
    title={Towards One-shot Neural Combinatorial Solvers: Theoretical and Empirical Notes on the Cardinality-Constrained Case}, 
    author={Runzhong Wang and Li Shen and Yiting Chen and Xiaokang Yang and Junchi Yan},
    year={2023},
    booktitle={ICLR}
}
```
