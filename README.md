The stock market is an extremely complex dynamic system. A remarkable number of works in literature has been devoted to understanding the stock market and developing effective trading strategies. In the paper titled *Practical Deep Reinforcement Learning Approach for Stock Trading* by Xiong et al., the authors examine the stock trading problem and proposes a deep reinforcement learning approach to tackle this challenging problem. In particular, the authors apply **Deep Deterministic Policy Gradient (DDPG)** to optimize for an effective stock trading strategy. 

The purpose of this project is to understand this paper, replicate its method and further explore the stock trading problem beyond the scope of this paper. We will first describe the problem of interest. Then we will discuss the method proposed in this paper and contrast it with existing alternatives. Next, we will describe the data used for training and testing, and explain how such data is processed. We include implementation details, followed by a discussion on model improvement and an alternative method to DDPG, called **Proximal Policy Optimization (PPO)**. We wrap up with experiment results and an evaluation.     

# Overview
*
* [Results](#results)
* [Data Repliation](#data-replication)


# Problem Description
In the aforementioned paper, the authors aim to develop a stock trading strategy with high profit. Suppose there is a pool of *D* stocks that we are interested in investing. A trading strategy consists of buying, selling and holding decisions for each of the *D* stocks at every time step, in response to the dynamic of stock market. The authors model the stock trading process using a **Markov Decision Process (MDP)**. We now set forth some notation and describe this model.

## Stock trading process as an MDP
TODO: insert diagram

**Agent**: investors

**Environment**: stock market

**State** (*s*): Each state s contains two vectors p and h, and a constant b. The vector p contains the prices of the D stocks. Vector h keeps track of how many of each stock we are holding. And the real number b is the balance we have available for buying more shares.

**Action** (*a*): Each action a encodes a set of actions on all the stocks. For each of the D stocks, we can sell some number of shares, buy some number of shares, or hold, which means do nothing about this stock.

**Reward** (*r*): is the change in the portfolio value given the last state s and action a, the portfolio value is price of the stocks we have, denoted by p, times the numbers of holding shares h plus the current balance b. 

Now the stock trading problem becomes a maximization problem of the expected total reward, which can be captured by the action-value function.

## Model refinements


# Methods
## Existing methods
Many other researchers have studied problems along the line of asset allocation. For example, the modern portfolio theory by Markowitz  goes back to the 50s. This is a traditional approach to the portfolio management problem, in which we are given a collection of investment options, like different stocks. We are interested in figuring out a way to assign weights to the available options, such that the expected return is maximized subject to a certain risk level. The optimization model in this approach utilizes the mean and covariance information of the past performance of the stocks. 

This approach has a few disadvantages: An observation is that a large amount of information is needed in this method, including the joint probability distribution among all the stocks, and a massive covariance matrix, things become intractable very quickly when the problem scale increases. Also people observed that it has stability issues. Basically when there’s a small change to the input, the optimal portfolio can change drastically. 

Another approach is to use Markov decision process to model the stock trading process, and then maximize the expected return using dynamic programming. From the solution we can then obtain an optimal policy. For value iteration and policy iteration, etc. the exact structure of the MDP, like the transition probabilities, are needed. This becomes a drawback when we don’t know how to make specifications to the Markov decision process. Also, this requirement restricts the tractable sizes of the state and action spaces, so this approach is not the best for large-scale problems. Sometimes we might not know the rewards either. So we need a better method especially for complex systems like the stock market.

## Method proposed in the paper
Now we give an overview of the method that the authors came up with for the stock trading problem. In particular, we describe the deep deterministic policy gradient algorithm that this whole study relies on. 

The high-level idea of the method in this paper is  to model the stock trading process with an MDP, which we have described in the first slide, and then to optimize the total expected reward by maximizing an action-value function. This optimization problem is quite challenging because the action value function is unknown to the decision maker. It has to be learned with feedbacks from the environment regarding different actions. This makes deep reinforcement learning a desirable approach. More specifically, this optimization problem is solved by an algorithm called deep deterministic policy gradient algorithm (DDPG).

## Details of DDPG
Next we focus on this DDPG algorithm. DDPG is modified from the deterministic policy gradient algorithm, and the authors adapted DDPG specifically to the MDP model for stock trading. 

The overall structure of DDPG is consistent with the policy gradient  framework that we’ve seen in the lectures. We also learned from the lectures that the crux of the policy gradient framework is to construct a good estimator for the gradient of the expectation function in the objective, usually denoted by capital J. In the general form of an estimator for its gradient, there is a Q term capturing THE VALUE resulted from some actions given states. And this is followed by  the gradient of log pi, which encodes information about decisions made on the actor’s end. These two portions can be approximated by deep neural networks, which is the actor-critic component in the PG framework. This figure is provided in the paper. This upper left network is the actor network. This mu maps the states to the actions, and it learns about how the agent selects actions. Here theta_mu is the set of network parameters of mu. N is a random process. Noises are sampled from N and added to the output of mu to broaden the scope of explorable actions. The network on the right is the critic network. This Q learns about the policy value of an action under a state, which in some way critiques the current policy that the agent adopts. Theta Q is the set of parameters in the critic network.

Algorithm overview TODO: insert an animation.

## Further exploration: PPO


# Data Acquisition and Processing
## Data used in the paper
The authors choose to use the Dow jones 30 stocks as the stocks of consideration. These stocks comprise 30 large companies, which are used to evaluate the Dow Jones Industrial Average. This is a commonly used index that reflects the overall performance of the US stock market. The authors had access to the daily prices of these chosen stocks from January the first in 2009 all the way up to September the 30th in 2018. The daily data from January the first in 2009 to the first day of 2016 is used for training and validation. Data after that up to September the 30th in 2018 was used for testing the trained agent’s performance. Yahoo finance is a wonderful source for alternative datasets. 

## Data used in this project
For this project, we used the provided minute-level volume weighted average prices. TODO:SPECIFY TRAINING AND TESTING SETUP This set of data happens to have many missing entries and NAN entries.  One company is omitted for too many missing dates. For consistency, we omit data during after-hours and stock market holidays. We noticed that some data is still missing, so we filled in with the nearest previous data.


# Implementation Details
------
INSTALL DEPENDENCIES:
-------
1) Install the following dependencies: 


- OpenAI Baselines:  https://github.com/openai/baselines

- OpenAI Gym:  https://github.com/openai/gym

2) If the packages is not installed in a default location, you need to add the following to your `~/.bash_profile` (edit or create the file named `bash_profile` under your home and add the following lines)

For example:
* ` export BASELINES_DIR="/Users/yourname/baselines" `
* ` export GYM_DIR="/Users/yourname/gym" `

3) Test the installation:
  ```
  pip install pytest
  cd $BASELINES_DIR
  pytest
  cd $GYM_DIR
  pytest
  ```
4) Test OpenAI Atari Pong game
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```
If this works, then it is ready to incorportate stock trading environment.

BUILD STOCK TRADING ENVIRONMENT
-----
1) Clone the repository

  ```
  git clone https://github.com/xinyi6666/stock_trading.git
  cd stock_trading
  ```

2) Make stock trading environment

- Make stock trading envs folder in Gym
  
  ```
  mkdir $GYM_DIR/envs/stocktrade
  cd $GYM_DIR/envs
  ```

- Copy `src/stock_trading_env.py`, `src/stock_trading_testenv.py`, `src/__init__.py` to `./stocktrade`

- Copy the following lines to `__init__.py` in the `$GYM_DIR/envs` folder:

```
register(
        id='StockTrade-v0',
        entry_point='gym.envs.stocktrade:StockTradingEnv',
        )

register(
        id='StockTradeTest-v0',
        entry_point='gym.envs.stocktrade:StockTradingTestEnv',
        )
```
EDIT BASELINES RUN FILE
----
Now we need to modify the `run.py` in the baselines
```
cd $BASELINES_DIR/baselines/
mv run.py run.bak
cp src/run.py ./
```
This will rename the original `run.py` to `run.bak` in the baseline folder and copy our customized `run.py` to baselines folder.

PROCESS DATA
----
If you do have the raw minute-level data from 2018/09/05 to 2021/02/17, you can use `src/data_process.py` to generate the data use for training and testing. If you do not have the minute-level data, or only have day-level data, please see section **GENERATING YOUR OWN DATA FROM ANY SOURCES**.

1) Locate the folder `historical_price` which contains the historical prices for companies
```
cd historical_price/
ls *.csv > list.txt 
```
Now in `list.txt`, it contains all file names with `.csv` extension in the folder. You can edit the list of companies you want to process by deleting some rows.

2) Edit `path` and `list_path` at the beginning of the `src/data_process.py` 

3) Process the data
```
cd src/
python data_process.py
```
It will generate a file called `clean_data.csv` in `src/`

4) Now create a Data folder for training and testing and put the processed data inside
```
mkdir $GYM_DIR/envs/stocktrade/Data
cp clean_data.csv $GYM/envs/stocktrade/Data/
```

GENERATING YOUR OWN DATA FROM ANY SOURCES (OPTIONAL)
----
1) If you don't have the correct raw data files, you need to modify the `data_process.py` file to write a `clean_data.csv` with a table in the following form:
```
            date   min      AAL      AAPL  ...      WFC       WMT      XOM      XRX
0       20180905   870  40.2246   56.4054  ...  58.9983   95.6207  80.2367  27.6172
1       20180905   871  40.0826   56.3734  ...  58.9704   95.6837  80.2245  27.6006
2       20180905   872  40.0226   56.3939  ...  58.9896   95.7038  80.2527  27.5923
3       20180905   873  39.9439   56.3964  ...  58.9831   95.7348  80.2153  27.5964
4       20180905   874  39.8317   56.3738  ...  58.9917   95.7342  80.2137  27.5994
          ...   ...      ...       ...  ...      ...       ...      ...      ...
```

2) Modify the `stop_min` in `stock_trading_env.py` and `start_min` in `stock_trading_testenv.py` according to your testing and training period.

3) Continue to the section [TRAIN AND TEST](#train-and-test)


TRAIN AND TEST
----
1) Edit `path` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to your local paths

2) (Optional) Edit `trade_interval` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to the number of minutes of trading interval you want

3) Go to baseline folder
```
cd /$BASELINES_DIR
```

### Train and Test with DDPG ###
Since save/load function in baseline does not work for DDPG, you can only train and then play one episode
```
python -m baselines.run --alg=ddpg --network=mlp --env=StockTrade-v0 --num_timesteps=2e6  --actor_lr=1.0e-5 
--critic_lr=1.0e-5 --gamma=1 --play
```
It will prompt lines for entering the names of log files to store the episode rewards and the replay portfolio values for each state.

### Train and Test with PPO ###
Train and save the network by runing
```
python -m baselines.run --alg=ppo2 --network=mlp --env=StockTrade-v0 --num_timesteps=5e6 --gamma=1 
--save_path=YOUR_SAVE_PATH
```
Testing:
```
python -m baselines.run --alg=ppo2 --network=mlp --env=StockTrade-v0 --num_timestep=0 --gamma=1 
--load_path=YOUR_LOAD_PATH --play
```

It will prompt lines for entering the names of log files to store the episode rewards and the replay portfolio values for each state.

# Results and Evaluation

# References
* Xiong, Z., Liu, X. Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement learning approach for stock trading. *arXiv preprint arXiv:1811.07522*.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
