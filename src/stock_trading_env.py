import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt

# Import training data
path = './'
training_data = pd.read_csv(path + '/Data/clean_data.csv').to_numpy()
total_row, total_col = training_data.shape

# This will read the reward log file name using stdin
log_file_name = input("Type Reward Log File Name:")
n_company = total_col - 2  # remove first two columns
initial_asset = 10000
final_reward = []  # for record the final reward

# rows to start and end
start_row = 0
end_row = 211806

trade_interval = 15

refined_model = True


class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    " Stock Trading Environment"
    """ Define constructor """

    def __init__(self, trading_interval=trade_interval):

        self.row = start_row

        # The interval for taking action is according to trading_interval
        self.trading_interval = trading_interval

        # Calculate a fix upper/lower bound according to the stock price at the beginning
        bound = min(np.mean(initial_asset / training_data[0][2:]), 100)

        # Set up the action space
        if not refined_model:
            self.action_space = spaces.Box(low=-bound, high=bound, shape=(n_company,), dtype=np.int)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(n_company,), dtype=np.double)

        # number of holding stock + prices for each companies + portfolio value
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(n_company * 2 + 1,))

        self.done = False
        self.beginning_asset = initial_asset
        self.total_asset = initial_asset

        if not refined_model:
            self.stock_price = training_data[self.row][2:]
            # number of holding stock + prices for each companies + portfolio value
            self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]
        else:
            self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]

        self.reward = 0
        self.total_reward_buffer = [0]

        self._seed()

    def reset(self):
        self.row = 0
        self.total_asset = initial_asset
        self.stock_price = training_data[self.row][2:]

        if refined_model:
            self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]
        else:
            self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]

        self.reward = 0
        self.total_reward_buffer = [0]
        return self.state

    def _buy_sell_stock(self, action):
        # Finding which indices are corresponding to buy action, and which are corresponding to sell action
        buy_idx = []
        if not refined_model:
            for i in range(n_company):
                if action[i] < 0:  # sell stock
                    n_sell = min(abs(action[i]), self.state[i])

                    # update current cash balance
                    self.state[-1] += self.stock_price[i] * n_sell

                    # update number of stocks stocking
                    self.state[i] -= n_sell

                if action[i] > 0:
                    buy_idx.append(i)

                # Calculate the total money required for buying all the stock
                total_money_required = sum(self.stock_price[buy_idx] * action[buy_idx])
                adjusted_action = action

                # If the agent does not have enough cash, then we need to adjust the number of stock buying depending on
                # how many cash it has
                if total_money_required > self.state[-1]:
                    adjusted_action[buy_idx] = np.floor(action[buy_idx] * (self.state[-1] / total_money_required))

                # Buy stocks
                for i in buy_idx:
                    n_buy = adjusted_action[i]

                    # Update the current cash balance
                    self.state[-1] -= self.stock_price[i] * n_buy

                    # update number of stocks stocking
                    self.state[i] += n_buy
        else:
            tol = 1.0e-4
            for i in range(n_company):
                if action[i] < -tol:  # sell stock
                    n_sell = min(abs(action[i]) * self.state[i], self.state[i])
                    self.state[-1] += self.stock_price[i] * n_sell
                    self.state[i] -= n_sell
                    assert self.state[i] >= 0
                if action[i] > tol:
                    buy_idx.append(i)

            adjusted_action = action
            if sum(action[buy_idx]) > (self.state[-1] / self.total_asset) and sum(action[buy_idx]) > tol:
                adjusted_action[buy_idx] = action[buy_idx] / sum(action[buy_idx])

            for i in buy_idx:
                if adjusted_action[i] > tol and self.state[-1] > self.stock_price[i]:
                    n_buy = (adjusted_action[i] * self.state[-1]) / self.stock_price[i]
                    self.state[-1] -= self.stock_price[i] * n_buy
                    self.state[i] += n_buy

    def step(self, actions):
        self.done = self.row + self.trading_interval >= end_row

        if self.done:
            final_reward.append(self.total_reward_buffer[-1])
            log = open("./" + log_file_name, 'a')
            log.write("%f\n" % self.total_reward_buffer[-1])
            log.close()
            print(self.total_reward_buffer[-1])
            plt.plot(final_reward, 'r')
            plt.savefig('training_curve.png')
            plt.close()

            plt.plot(self.total_reward_buffer, 'r')
            plt.savefig('portfolio_curve.png')
            plt.close()

            return self.state, 0, self.done, {}
        else:
            self._buy_sell_stock(actions)

            # Update the row number
            self.row += self.trading_interval
            stock_price_old = self.stock_price
            self.stock_price = training_data[self.row][2:]
            self.total_asset = self.state[-1] + sum(self.stock_price*self.state[0:n_company])

            # Update current state
            if refined_model:
                change_ratio = (self.stock_price - stock_price_old) / stock_price_old * 100
                self.state = list(self.state[0:n_company]) + change_ratio.tolist() + [self.state[-1]]
            else:
                self.state = list(self.state[0:n_company]) + self.stock_price.tolist() + [self.state[-1]]

            # Calculate the total portfolio value

            total_reward = self.total_asset - self.beginning_asset

            # Current reward will be the change of portfolio value, it is the same as the change of reward
            self.reward = total_reward - self.total_reward_buffer[-1]

            # Record the reward change
            self.total_reward_buffer.append(total_reward)

            return self.state, self.reward, self.done, {}

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



