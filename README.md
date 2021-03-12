The stock market is an extremely complex dynamic system. A remarkable number of works in the literature has been devoted to understanding the stock market and developing effective trading strategies. In the paper titled *Practical Deep Reinforcement Learning Approach for Stock Trading* by Xiong et al., the authors examine the stock trading problem and apply a deep reinforcement learning approach to tackle this challenging problem. In particular, the authors apply **Deep Deterministic Policy Gradient (DDPG)** to optimize for an effective stock trading strategy. 

The purpose of this project is to understand this paper, replicate its method and further explore the stock trading problem beyond the scope of this paper. We will first describe the problem of interest formulated as a Markov Decision Process model, followed by a discussion on model improvement. Then we will discuss the method proposed in this paper and contrast it against existing alternatives. We will highlight DDPG and its alternative called **Proximal Policy Optimization (PPO)** which we explored beyond this paper. Next, we will describe the data used for training and testing, and explain how such data is processed. We include implementation details and code execution instructions. We wrap up with the experiment results, the method evaluation and some concluding remarks. Below is an overview of the succeeding discussions.     

# Overview
* [Problem Description](#problem-description)
  * [Stock trading process as an MDP](#stock-trading-process-as-an-MDP)
  * [Model refinements](#model-refinements)
* [Methods]
  * [Existing methods](#existing-methods)
  * [Method proposed in the paper](#method-proposed-in-the-paper)
  * [DDPG](#deep-deterministic-policy-gradient-(DDPG))
  * [PPO](#Proximal-Policy-Optimization-(PPO))
* [Code Explanation](#code-explaination)
* [Results](#results)
* [Data Replication](#data-replication)


# Problem Description
In the aforementioned paper, the authors aim to develop a stock trading strategy with high profit. Suppose there is a pool of *D* stocks that we are interested in investing. A trading strategy consists of buying, selling and holding decisions for each of the *D* stocks at every time step, in response to the dynamic of stock market. The authors model the stock trading process using a **Markov Decision Process (MDP)**. We now set forth some notation and describe the MDP model.

## Stock trading process as an MDP
An MDP is a discrete-time stochastic process. In this problem context, the MDP  can be described using the following components.  

**Agent**: investors.

**Environment**: stock market.

**State** (*s*): At each time step, *s* contains three pieces of information (1) a vector *p* with the prices of the *D* stocks; (2) a vector *h* representing how many shares of each of the *D* stocks that the agent owns; (3) a constant *b* that keeps track of the amount of money available for future investments. 

**Action** (*a*): For each of the *D* stocks, the agent can sell *k* shares, buy *k* shares, or hold, where *k* is a positive integer. Hold means that no shares  of a given stock is bought or sold. We note that this action space is discrete. 

**Reward** (*r*): At each time step, the reward is the change in the portfolio value given the last state *s*, the current state *s'* and action *a*. The portfolio value is the summation of *b* and stock prices (*p*) multiplied by the corresponding numbers of shares (*h*) held by the agent. 

Now the best stock trading strategy is a sequence of actions dependent on the states that maximizes the total expected reward. 

## Model refinements
For our project, we made a few adjustments to the MDP model given in the paper (described above). More specifically, we re-defined the states and actions, so that the model works better with a deep reinforcement learning approach. 

**State**: At each time step, the state still contains three pieces of information. The change is made to the first piece. We replace the vector of all *D* stock prices (*p*), by a vector of price increment rates. Such rate for each stock at every time step is computed by (*current price - previous price*)/*previous price*.
<p align="center">
<img width="360" src="https://user-images.githubusercontent.com/17188583/110899203-4f452880-82c6-11eb-9d61-85001a14d33d.png">
</p> 

The benefits of this modification include: 
* The agent receives valuable information about the short-term trends of stock price changes.
* We will likely not observe the same stock price multiple times after a long period of time. Therefore, in a deep reinforcement learning approach, this adjustment avoids the agent from encountering too many states that were not explored in the training phase. (We will explain the deep reinforcement learning approach later in the methods section.)

**Action**:We still have three actions-buy, sell and hold. However, we now make the action space continuous. The number of shares that we buy or sell now depends on a number *a* in the continuous interval *[-1,1]*.
<p align="center">
  <img width="480" src="https://user-images.githubusercontent.com/17188583/110899232-5a985400-82c6-11eb-919a-08cb396419dd.png">
</p>

The benefits of this modification include:
* The agent can now buy or sell fractional shares, which is a trading option explored in many other works. 
* The paper applies the DDPG algorithm to maximize the action-value function, which we will explain in the methods section. One crucial advantage of DDPG is that it can handle continuous action spaces. 

Next we discuss existing methods for the stock trading problem, and provide an overview of the method proposed in the paper. 



# Methods
In this section, we first describe and evaluate two earlier methods for the stock trading problem. Then we discuss the method proposed in the paper and highlight the key algorithm--DDPG. After that, we give an overview of PPO, which is an alternative algorithm to DDPG that we explored beyond the assigned paper.

## Existing methods
Many other researchers have studied the stock trading problem. For example, the Modern Portfolio Theory and Markowitz Model were proposed in the 50s. This is a traditional approach to the portfolio management problem. In this problem, we are given a collection of investment options, such as different stocks. The goal is to assign proper weights to the available investment options, such that the expected total return is maximized, subject to some risk constraint. This approach utilizes the mean and covariance information of the past performance of the stocks, which leads to a disadvantage--a large amount of information is needed, including the joint probability distribution and a covariance matrix. They become intractable very quickly as the problem scale increases. People have also observed stability issues. When there is a small change in the input, the optimal portfolio returned by this method can change drastically. 

Another approach is to use MDP to model the stock trading process, and then maximize the expected total reward using dynamic programming. Value iteration and policy iteration require the transition model to be known. This becomes a drawback when we do not know how to make specifications for the MDP. Also, this requirement restricts the tractable sizes of the state and action spaces. Therefore, this approach is also limited for large-scale problems. 

The drawbacks of these two methods can be overcome with a deep reinforcement learning approach. 

## Method proposed in the paper
We now go over the method adopted by the authors for the stock trading problem. In particular, we highlight the DDPG algorithm that the paper relies on. 

The high-level idea of the method in this paper is to model the stock trading process with an MDP (described in an earlier section), and then to optimize the expected total reward by maximizing an action-value function. The underlying challenge is that the action-value function is unknown to the agent, so it has to be learned from the feedbacks given by  the environment regarding different actions. This makes deep reinforcement learning a desirable approach. In the paper, this optimization problem is solved by DDPG.

## Deep Deterministic Policy Gradient (DDPG)
Next we focus on the DDPG algorithm. DDPG is a variant of the deterministic policy gradient algorithm. The authors adapted DDPG specifically to the MDP model for stock trading. We know that the crux of the policy gradient framework is to construct a good estimator for the gradient, which usually comprises of a value (Q) term and an actor (gradient of log policy evaluation) term. DDPG approximates these two terms with two deep neural networks. called the actor network and the critic network. 

<p align="center">
  <img width="480" src="/fig/actor_critic.png">
</p>

This actor network takes in the current state and outputs an action. We note that the action space can be continuous. The critic network takes this action and the current state as input, and estimates the value (Q). This estimated value then critiques the actor’s decision. 
<p align="center">
  <img width="640" src="/fig/DDPG_algorithm.png">
</p>
The figure above is the algorithm description provided in the paper. We created the animation below to help visualize the process. The flowchart shows the essential components in this algorithm, namely the actor and critic networks, the target actor and target critic networks, and a replay buffer. We next go over the DDPG algorithm; the green highlights in the animation correpond to the components involved in each step. 

<p align="center">
  <img width="640" src="/fig/DDPG_overview.gif">
</p>

1. We first initialize the actor and critic networks, the target networks and a replay buffer. 
2. For every episode, we take the initial state and instantiate a random process *N* to generate noise for actions from the actor network. The noise allows us to explore more actions. We then loop over the time steps *t* and do the following.
3. For the state at *t*, we obtain a noise-modified action from the actor network. Next we execute this action to get a reward and a new state. We add this chunk of new history to the replay buffer, from which we sample a mini batch.
4. With this mini batch and outputs from the target networks, we optimize for a new critic network parameter and update the critic network. 
5. Again using this mini batch, we use the actor and critic networks to get the sampled policy gradient, and update the actor policy. 
6. Lastly we update the target networks and continue with the loop.  

The replay buffer and the target networks are two technical tricks. Replay buffer reduces the temporal correlation of the simulated trajectories, whereby lowering the variance of estimations. The target networks regularize the learning algorithms of the actor network and the critic network. It has been observed that if we directly use the gradient from mini batch samples, these learning algorithms could diverge.


## Proximal Policy Optimization (PPO)
We explored another method that works well with continuous action space, called PPO. It also falls under the policy gradient framework for reinforcement learning. This method has similar benefits as the trust region policy optimization (TRPO). There are two versions of PPO surrogate objective functions. One version involves a KL penalty term. We consider the other version stated below, for its ease of implementation.  

<p align="center">
  <img width="480" src="https://user-images.githubusercontent.com/17188583/110910543-1feae780-82d7-11eb-8958-4398cf97f2bd.png">
</p>

PPO maximizes this objective to acquire the optimal parameter theta. 

We next provide some intuitions behind this objective function. The estimated advantage term *A* in this objective is an approximation of the difference between the action value of some action and the expected reward to go under a given state. This difference informs us whether the outcome of an action is better or worse than average, so that we can increase or decrease the weight for this action accordingly. The *r* term represents a ratio that measures how different two policies are. As we run gradient descent over limited batches of past experiences, the policy parameters could be pushed far enough to ruin the policy. The ratio and the clip operation adjust the estimated advantage, such that the new policy will not be too far from the previous policy. 


# Data Acquisition and Processing
## Data used in the paper
The authors choose the Dow Jones 30 stocks as the stocks of consideration. These stocks comprise 30 large companies, which are used to evaluate the Dow Jones Industrial Average. This is a commonly used index that reflects the overall performance of the US stock market. The authors had access to the daily prices of these chosen stocks from January 1, 2009 to September 30, 2018. The daily data from January 1, 2009 to January 1, 2016 is used for training and validation. Data after that up to September 30, 2018 was used for testing the performance of the trained agent. 

## Data used in this project
For this project, we used the provided minute-level volume weighted average stock prices. This set of data happens to have many missing entries and NAN entries. The stock of one company is omitted for too many missing dates, so we considered 85 stocks in total. In addition, we only utilize data during trading hours on trading days (days that are not stock market holidays) for consistency. To deal with the data entries that are still missing, we fill them in with the nearest previous price data. The minute-level prices from September 5, 2018 to November 1, 2020 were used to train the agent. The data after November 1, 2020 up to February 17, 2021 was used for testing. More details of data processing is included in the implementation section.



# Code Explanation
In this section, we are going to explain how to implement a cumsom gym enviroment for the stock trading. This basically explains what are included in `stock_trading_env.py` and `stock_trading_testenvs.py`. We have two versions of stock trading environments which are corresponding to the model in the original paper and the refined model, respectively. 

First before talking into details, the environement should contains all necessary functionaility to run an agents and allow it to learn. Each environment must implement in the following form:
```
import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...
```
For more explanation, please refer to this [blog](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e).


## Model in the paper
Before making the custom gym environment, we need to define some variables and import data which are going to be used over the stock trading environment
```
import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces

# Import training data
path = './'
training_data = pd.read_csv(path+'/Data/clean_data.csv').to_numpy()
total_row, total_col = training_data.shape

# This will read the reward log file name using stdin 
log_file_name = input("Type Reward Log File Name:")
n_company = total_col - 2  # remove first two columns
initial_asset = 10000
final_reward = [] # for record the final reward 
trade_interval = 15

# rows to start and end 
start_row = 0
end_row = 211806

```

### Constructor 
Now we are going to create constructor
```

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, trading_interval = trade_interval):
        self.row = start_row
        
        # The interval for taking action is according to trading_interval
        self.trading_interval = trading_interval  
        
        # Calculate a fix upper/lower bound according to the stock price at the beginning
        bound = min(np.mean(initial_asset / training_data[0][2:]), 100)
        
        # Set up the action space
        self.action_space = spaces.Box(low=-bound, high=bound, shape=(n_company,), dtype=np.int)

        # number of holding stock + prices for each companies + portfolio value 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(n_company * 2 + 1,))
        
        self.stock_price = training_data[self.row][2:]
        self.done = False
        self.beginning_asset = initial_asset
        self.total_asset = initial_asset
      
        
        # number of holding stock + prices for each companies + portfolio value 
        self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]
        self.reward = 0
        self.total_reward_buffer = [0]

        self._seed()    
    
```

### Reset the environment
```
    def reset(self):
        self.row = start_row
        self.total_asset = initial_asset
        self.stock_price = training_data[self.row][2:]
        self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]
        self.reward = 0
        self.total_reward_buffer = [0]
        return self.state
```


### Buy and sell the stocks
Before defining the step method, we are going to define a private method called `_buy_sell_stock` for determing what is the next state. First we need to determine which indices in the input action is corresponding to buy/sell/hold. Meanwhile, we can perform the sell action and update the current cash balance and the number of stock held in the portfolio. 
```
    def _buy_sell_stock(self, action):
    
        # Finding which indices are corresponding to buy action, and which are corresponding to sell action
        buy_idx = []
        for i in range(n_company):
            if action[i] < 0: # sell stock 
                n_sell = min(abs(action[i]), self.state[i])
                
                # update current cash balance 
                self.state[-1] += self.stock_price[i] * n_sell 
                
                # update number of stocks stocking 
                self.state[i] -= n_sell
                
            if action[i] > 0:
                buy_idx.append(i)
```
After that, we perform sell action. Since the the total amount of cash required for buying all the stocks may be greater than the total amount of cash the agent has. We need to adjust the action by using the following formula:

<H2 align="center">  <img src="https://user-images.githubusercontent.com/47408833/110899074-11480480-82c6-11eb-8472-e1cd2bbf2c67.png" width="500"></H2>

```
        # Calculate the total money required for buying all the stock
        total_money_required = sum(self.stock_price[buy_idx] * action[buy_idx])
        adjusted_action = action
        
        # If the agent does not have enough cash, then we need to adjust the number of stock buying depending on
        # how many cash it has
        if total_money_required > self.state[-1]:
            adjusted_action[buy_idx] = np.floor(action[buy_idx] * (self.state[-1] / total_money_required))
```
After calculation, we buy the corresponding stocks and update the number of stocks held and cash balance in the state.

```
        # Buy stocks
        for i in buy_idx:
            n_buy = adjusted_action[i]
            
            # Update the current cash balance
            self.state[-1] -= self.stock_price[i] * n_buy  
            
             # update number of stocks stocking 
            self.state[i] += n_buy
```

### Step method
Now we are going to create the step method, which should return what is the reward for stepping to new state, next state, and if the agent is at the end of one episode. First, we determine if one episode is going to end:
```
    def step(self, actions):
        
        self.done = self.row+self.trading_interval >= stop_row
```
If the episode should be ending, then we record the final portfolio value to the log file 
```
        if self.done:
            final_reward.append(self.total_reward_buffer[-1])
            log = open("./"+log_file_name,'a')
            log.write("%f\n"%self.total_reward_buffer[-1])
            log.close()
            
            return self.state, 0, self.done, {}
```
Otherwise, we calculate the part of the state by calling `_buy_sell_stock`. After that, we are going to update the stock price part in the state by using the updated price data. In addition, we also calculate the total portfolio value, the reward and store them in the buffer.
```
        else:
            self._buy_sell_stock(actions)
            
            # Update the row number 
            self.row += self.trading_interval
            
            self.stock_price = training_data[self.row][2:]
            
            # Update current state 
            self.state = list(self.state[0:n_company]) + self.stock_price.tolist() + [self.state[-1]]
            
            # Calculate the total portfolio value 
            self.total_asset = self.state[-1] + sum(self.stock_price*self.state[0:n_company])
            
           total_reward = self.total_asset- self.beginning_asset
            
            # Current reward will be the change of portfolio value, it is the same as the change of reward
            self.reward = total_reward - self.total_reward_buffer[-1]
            
            #  Record the reward change 
            self.total_reward_buffer.append(total_reward)

            return self.state, self.reward, self.done, {}
```

### Other methods
 
```
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
```

## Refined Model

Similar to the previous section, we need to import data and define variables. Since lots of codes are repeated, we do not repost the them here, and will only note the change in each section. For detailed code (without the changes in this section) for `__init__`, `reset`,`render` and `_seed` methods, please refer to section [Model in the Paper](#model-in-the-paper)

### Constructor 
Since we have changed the the action space to continuous from -1 to 1, here we change the defination of action space to 
```
self.action_space = spaces.Box(low=-1, high=1, shape=(n_company,), dtype=np.double)
```

Now for the state space, we replace the price of stock to 0 at declaration:

```
self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]
```


### Reset the environment
Similarly to the previous section, in `reset` method, we need to change the part for reseeting `self.state` to 
```
self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]
```

### Buy and sell the stocks
For the `_buy_sell_stock` method, there are lots of changes. By that, here we repost the detailed code and explain what is doing. Since the action returned is continuous, then we need to have a tolerance for determining if action value is numerically zero. After that, we are doing the similar process for finding out indices corresponding to buy and sell. For calculating the number of stocks selling, we determine that use action value times the current stock holding.
```
    def _buy_sell_stock(self, action):
    
        buy_idx = []
        tol = 1.0e-4 # define tolerance for determing if a action is buy/sell/hold
        for i in range(n_company):
            if action[i] < -tol:  # sell stock
                n_sell = min(abs(action[i])*self.state[i], self.state[i])
                self.state[-1] += self.stock_price[i] * n_sell
                self.state[i] -= n_sell
                assert self.state[i] >= 0
            if action[i] > 1.0e-4:
                buy_idx.append(i)
```
After that, we perform sell action. We now use the formula
<H3 align="center">  <img src="https://user-images.githubusercontent.com/47408833/110911669-9cca9100-82d8-11eb-986d-30dfb51f6112.png", width="400"> </H3>
to adjust the action for buying the stock if we don't have enough cash. After calculation, we buy the corresponding stocks and update the number of stocks held and cash balance in the state.

```
         adjusted_action = action
         if sum(action[buy_idx]) > (self.state[-1] / self.total_asset) and sum(action[buy_idx]) > tol:
               adjusted_action[buy_idx] = action[buy_idx] / sum(action[buy_idx])
               
        # Buy stocks
        for i in buy_idx:
            if adjusted_action[i] > tol and self.state[-1] > self.stock_price[i]:
                n_buy = (adjusted_action[i] * self.state[-1]) / self.stock_price[i]
                self.state[-1] -= self.stock_price[i] * n_buy
                self.state[i] += n_buy
```

### Step method
For the step method, we only need to change how we calculate `self.state`. In order to do that, we need to calculate the change of the stock price in percentage. 
```
        change_ratio = (self.stock_price - stock_price_old) / stock_price_old * 100
        self.state = list(self.state[0:n_company]) + change_ratio.tolist() + [self.state[-1]]
```


For completing the code, we also need to include `render` and `_seed` described in section [Other methods](#other-methods)

## Test Environment
The overall implementation for the test environment is basically the same. Only things we have to change is `start_row`, `end_row` at the variable declaration, and the name for the class to `StockTradingTestEnv`.




# Results and Evaluation
Recall that we have two MDP models for the stock trading process--the original model introduced in the paper, and the refined model with adjustments in the states and actions that we proposed. Now we present the results obtained using both the original model and the refined model in the figure below. 

<p align="center">
  <img width="480" src="/fig/old_vs_new_model.png">
</p>

We set the trading interval for training and testing to be 30 minutes. In other words, an agent takes one action every 30 minutes. The initial portfolio value is set at $10,000. After training the agents based on the two models, we run the policies on the testing data and plot the corresponding growth of total reward. In the figure, the x-axis shows the trading dates, and the y-axis is the portfolio value in dollars. The blue curve is the portfolio growth of the policy trained with the refined model, and the red curve corresponds to the policy obtained with the model proposed in the paper. Our refined model outperforms the original model. As shown in this figure, the blue curve is consistently higher than the red curve; by the end of the testing period, the policy derived from the refined model almost doubled the profit of the original model. 


The next figure captures the testing results of DDPG and PPO. Identical to the last set of experiments, we selected 30 minutes as the trading interval. In the figure below, the blue curve represents the portfolio change according to the policy found by DDPG, and the red curve is that of PPO. 

<p align="center">
  <img width="480" src="/fig/results.png">
</p>

Both policies attain similar portfolio values across all the trading dates. The two curves almost overlap over the first half of the trading dates. Toward the end of testing, the two portfolio values differ by only a few hundreds of dollars. 

<p align="center">
  <img width="300" src="/fig/table.png">
</p>

Lastly, we compare the DDPG and PPO results with SPY500 index and QQQ, whose components are the stocks of Nasdaq top 100 companies. More specifically, we evaluate these methods in terms of the return, the standard deviation of excess return, and Sharpe ratios. From the table above, we observe that both policies trained by the deep reinforcement learning algorithms beat the market index by over 10% in returns. Although their standard deviations are larger, their Sharpe ratios are comparable with the market index. Such results suggest that deep reinforcement learning approaches have great potentials in tackling the stock trading problem. 

# How to Use This Repository
## Install dependencies
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

## Build stock trading environment

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
## Edit baselines run file

Now we need to modify the `run.py` in the baselines
```
cd $BASELINES_DIR/baselines/
mv run.py run.bak
cp src/run.py ./
```
This will rename the original `run.py` to `run.bak` in the baseline folder and copy our customized `run.py` to baselines folder.

## Process data

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

## Generating your own data from any sources (optional) 

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


## Training and testing 

1) Edit `path` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to your local paths

2) (Optional) Edit `trade_interval` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to the number of minutes of trading interval you want

3) Go to baseline folder
```
cd /$BASELINES_DIR
```

### Train and test with DDPG ###
Since save/load method in baseline does not work for DDPG, you can only train and then play one episode
```
python -m baselines.run --alg=ddpg --network=mlp --env=StockTrade-v0 --num_timesteps=2e6  --actor_lr=1.0e-5 
--critic_lr=1.0e-5 --gamma=1 --play
```
It will prompt lines for entering the names of log files to store the episode rewards and the replay portfolio values for each state.

### Train and test with PPO ###
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


# Conclusion
In this project, we successfully replicated the method from *Practical Deep Reinforcement Learning Approach for Stock Trading* by Xiong et al. for the stock trading problem. We further adjusted the MDP model of the stock trading process, which improves the trading strategy obtained via DDPG. Moreover, we went beyond the paper and experimented with an alternative method called PPO. Although DDPG and PPO achieved similar results, it is worth noting that PPO trains the agent faster and is less sensitive than DDPG to the choice of learning rate. Other extensions from this paper can be explored in the future, such as trying other alternatives of DDPG (e.g. TRPO), and enriching the model by including additional factors (e.g. volumes, technical indicators). 

# References
* Xiong, Z., Liu, X. Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement learning approach for stock trading. *arXiv preprint arXiv:1811.07522*.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
