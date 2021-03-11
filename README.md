# STOCK TRADING
-------

# DATA REPLICATION
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


TRAIN AND TEST
----
1) Edit `path` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to your local paths

2) (Optional) Edit `trade_interval` in `$GYM_DIR/envs/stocktrade/stock_trading_env.py` and `$GYM_DIR/envs/stocktrade/stock_trading_testenv.py` to the number of minutes of trading interval you want

3) Go to baseline folder
```
cd /$BASELINEs_DIR
```

### Train and Test with DDPG ###
Since save/load function in baseline does not work for DDPG, you can only train and then play one episode
```
python -m baselines.run --alg=ddpg --network=mlp --env=StockTrade-v0 --num_timesteps=2e6  --actor_lr=1.0e-5 
--critic_lr=1.0e-5 --gamma=1 --play
```

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
GENERATING YOUR OWN DATA FROM ANY SOURCES (OPTIONAL)
----


