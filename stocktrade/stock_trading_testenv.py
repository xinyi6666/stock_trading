import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt

path = '/home/lby/virt_env_python3.6/lib/python3.6/site-packages/gym/envs/stocktrade'
#training_data = pd.read_csv(path+'/Data/clean_data.csv')
#training_data = training_data.iloc[:,range(32)].to_numpy()
training_data = pd.read_csv(path+'/Data/clean_test_data.csv').to_numpy()
#print(training_data)
log_file_name = input("Type Replay Log File Name:")
total_row, total_col = training_data.shape

n_company = total_col - 2  # remove first two columns
initial_asset = 10000
final_reward = []


class StockTradingTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    " Stock Trading Environment"
    """ Define constructor """

    def __init__(self, trading_interval = 30):
        self.row = 0
        self.trading_interval = trading_interval
#        bound = min(np.mean(initial_asset / training_data[0][2:]), 20)
#        self.action_space = spaces.Box(low=-bound, high=bound, shape=(n_company,), dtype=np.int)
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_company,), dtype=np.double)

        # portfolio value + prices for each companies + holding stock number
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(n_company * 2 + 1,))

        self.stock_price = training_data[self.row][2:]
        self.done = False
        self.beginning_asset = initial_asset
        self.total_asset = initial_asset

        #self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]
        self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]
        self.reward = 0
        self.total_reward_buffer = [0]

        self.reset()
        self._seed()

    def reset(self):
        self.row = 0
        self.total_asset = initial_asset
        self.stock_price = training_data[self.row][2:]
        self.state = [0 for _ in range(n_company)] + [0 for _ in range(n_company)] + [initial_asset]
        #self.state = [0 for _ in range(n_company)] + self.stock_price.tolist() + [initial_asset]
        self.reward = 0
        self.total_reward_buffer = [0]
        return self.state

    def _buy_sell_stock(self, action):
        buy_idx = []
        #print(action)
#        action /= 100
        
#        for i in range(n_company):
#            if action[i] < 0: # sell stock 
#                n_sell = min(abs(action[i]), self.state[i])
#                self.state[-1] += self.stock_price[i] * n_sell
#                self.state[i] -= n_sell
#                assert self.state[i] >= 0
#            if action[i] > 0:
#                buy_idx.append(i)

        for i in range(n_company):
            if action[i] < -1.0e-4:  # sell stock
                n_sell = min(abs(action[i])*self.state[i], self.state[i])
                self.state[-1] += self.stock_price[i] * n_sell
                self.state[i] -= n_sell
                assert self.state[i] >= 0
            if action[i] > 1.0e-4:
                buy_idx.append(i)
        
        
        adjusted_action = action
        if sum(action[buy_idx])>(self.state[-1]/self.total_asset) and sum(action[buy_idx])>1.0e-4:
            adjusted_action[buy_idx] = action[buy_idx]/sum(action[buy_idx])

        for i in buy_idx:
            if adjusted_action[i] > 1.0e-4 and self.state[-1] > self.stock_price[i]:
                n_buy = (adjusted_action[i]*self.state[-1])/self.stock_price[i]
                self.state[-1] -= self.stock_price[i] * n_buy
                self.state[i] += n_buy


#        total_money_required = sum(self.stock_price[buy_idx] * action[buy_idx])
#        adjusted_action = action
#
#        if total_money_required > self.state[-1]:
#            adjusted_action[buy_idx] = np.floor(action[buy_idx] * (self.state[-1] / total_money_required))
#
#        for i in buy_idx:
#            assert adjusted_action[i] >= 0
#            n_buy = adjusted_action[i]
#            self.state[-1] -= self.stock_price[i] * n_buy
#            self.state[i] += n_buy

    def step(self, actions):
        self.done = self.row+self.trading_interval >= total_row
        #self.done = self.row >= 80000

        if self.done:
            final_reward.append(self.total_reward_buffer[-1])
#            print(training_data[self.row][0])
#            print(training_data[self.row][1])
            print(self.total_reward_buffer[-1])
            plt.plot(self.total_reward_buffer,'r')
            plt.savefig('Testing_Result.png')
            plt.close()

            return self.state, self.reward, self.done, {}
        else:
            self._buy_sell_stock(actions)
            self.row += self.trading_interval
            stock_price_old = self.stock_price
            self.stock_price = training_data[self.row][2:]
            change_ratio = (self.stock_price-stock_price_old)/stock_price_old*100
            self.total_asset = self.state[-1] + sum(self.stock_price*self.state[0:n_company])
            self.state = list(self.state[0:n_company]) + change_ratio.tolist() + [self.state[-1]]
            total_reward = self.total_asset - self.beginning_asset
            self.reward = total_reward-self.total_reward_buffer[-1]
#            self.reward = total_reward#-self.total_reward_buffer[-1]
            self.total_reward_buffer.append(total_reward)
            log = open("./"+log_file_name,'a')
            log.write("%f\n"%self.total_asset)
            log.close()

            return self.state, self.reward, self.done, {}
            #self._buy_sell_stock(actions)
            #self.row += self.trading_interval
            #self.stock_price = training_data[self.row][2:]
            #self.total_asset = self.state[-1] + sum(self.stock_price*self.state[0:n_company])
            #self.state = list(self.state[0:n_company]) + self.stock_price.tolist() + [self.state[-1]]
            #total_reward = self.total_asset - self.beginning_asset
            #self.reward = total_reward-self.total_reward_buffer[-1]
#           # self.reward = total_reward#-self.total_reward_buffer[-1]
            #self.total_reward_buffer.append(total_reward)

            #return self.state, self.reward, self.done, {}

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




