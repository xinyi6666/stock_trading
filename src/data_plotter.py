import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

test_date = pd.read_csv('./test_date.log')
interval = int(len(test_date) / 5)
test_date_idx = []
i = 0
treasury_bill_rate = 0.13
while i < len(test_date):
    test_date_idx.append(datetime.datetime.strptime(
        str(test_date['date'][i]), '%Y%m%d').strftime(('%m/%d/%Y')))
    i += interval


def plot_ppo_ddpg(replay_ppo, replay_ddpg, min):
    qqq = pd.read_csv('./QQQ.csv')
    qqq_close = qqq['Close']
    qqq_close = qqq_close * 10000 / qqq_close[0]
    daily_return_qqq = (qqq_close[0:].to_numpy() - 10000) / 10000 * 100
    std_qqq = np.std(daily_return_qqq)
    return_perc_qqq = (qqq_close[71] - qqq_close[0]) / qqq_close[0] * 100
    sharpe_ratio_qqq = (return_perc_qqq - treasury_bill_rate) / std_qqq
    print("QQQ std is %9f" % std_qqq)
    print("QQQ return percentage is %9f" % return_perc_qqq)
    print("QQQ sharpe ratio is %9f" % sharpe_ratio_qqq)

    spy = pd.read_csv('./SPY.csv')
    spy_close = spy['Close']
    spy_close = spy_close * 10000 / spy_close[0]
    daily_return_spy = (spy_close[0:].to_numpy() - 10000) / 10000 * 100
    plt.plot(daily_return_spy, 'r')
    plt.savefig('spy_daily_return.pdf')
    plt.close()
    std_spy = np.std(daily_return_spy)
    return_perc_spy = (spy_close[71] - spy_close[0]) / spy_close[0] * 100
    sharpe_ratio_spy = (return_perc_spy - treasury_bill_rate) / std_spy

    print("SPY std is %9f" % std_spy)
    print("SPY return percentage is %9f" % return_perc_spy)
    print("SPY sharpe ratio is %9f" % sharpe_ratio_spy)

    ddpg_replay = pd.read_csv(replay_ddpg)
    ppo_replay = pd.read_csv(replay_ppo)
    daily_return_ddpg = (ddpg_replay[0:].to_numpy() - 10000) / 10000 * 100
    std_ddpg = np.std(daily_return_ddpg)
    daily_return_ppo = (ppo_replay[0:].to_numpy() - 10000) / 10000 * 100
    std_ppo = np.std(daily_return_ppo)
    return_perc_ppo = ((ppo_replay.to_numpy()[-1] - 10000) / 10000) * 100
    return_perc_ddpg = ((ddpg_replay.to_numpy()[-1] - 10000) / 10000) * 100
    sharpe_ratio_ppo = (return_perc_ppo - treasury_bill_rate) / std_ppo
    sharpe_ratio_ddpg = (return_perc_ddpg - treasury_bill_rate) / std_ddpg
    print("DDPG std is %9f" % std_ddpg)
    print("DDPG return percentage is %9f" % return_perc_ddpg)
    print("DDPG sharpe ratio is %9f" % sharpe_ratio_ddpg)

    print("PPO std is %9f" % std_ppo)
    print("PPO return percentage is %9f" % return_perc_ppo)
    print("PPO sharpe ratio is %9f" % sharpe_ratio_ppo)

    plt.plot(ddpg_replay.to_numpy(), 'b')
    plt.plot(ppo_replay.to_numpy(), 'r')
    if min == '1':
        plt.title('Testing Result ' + min + 'min')
    else:
        plt.title('Testing Result ' + min + 'mins')
    plt.ylabel('Portfolio Value in $')
    plt.xlabel('Trading Date')
    if min == '30':
        plt.xticks(np.arange(0, 1000, 200), labels=test_date_idx)
    elif min == '15':
        plt.xticks(np.arange(0, 2000, 400), labels=test_date_idx)
    elif min == '1':
        plt.xticks(np.arange(0, 2000 * 15, 400 * 15), labels=test_date_idx)
    plt.legend(["DDPG", "PPO"], loc="lower right")

    if min == '1':
        plt.savefig('Testing_result ' + min + ' min.pdf')
    else:
        plt.savefig('Testing_result ' + min + ' mins.pdf')

    plt.close()


def plot_new_old_model():
    ddpg_replay = pd.read_csv('replay_ddpg_30mins_old_model.log')
    ddpg_old = pd.read_csv('replay_ddpg_30mins_old_model.log')
    plt.plot(ddpg_replay.to_numpy(), 'b')
    plt.plot(ddpg_old.to_numpy(), 'r')
    plt.title('Comparison of Old and New Models')
    plt.ylabel('Portfolio Value in $')
    plt.xlabel('Trading Date')
    plt.xticks(np.arange(0, 1000, 200), labels=test_date_idx)
    plt.legend(["new", "old"], loc="lower right")
    plt.savefig('Comparison.pdf')
    plt.close()


def plot_training(train_log, type, min):
    train_data = pd.read_csv(train_log)
    plt.plot(train_data.to_numpy(), 'r')

    if min == '1':
        plt.title('Training Curve for ' + type + ' ' + min + 'min')
    else:
        plt.title('Training Curve for ' + type + ' ' + min + 'mins')

    plt.ylabel('Training Reward')
    plt.xlabel('Episode')

    if min == '1':
        plt.savefig('Training_' + type + '_' + min + '_min.pdf')
    else:
        plt.savefig('Training_' + type + '_' + min + '_mins.pdf')
    plt.close()


plot_ppo_ddpg('replay_ppo_30mins_new_model.log', 'replay_ddpg_30mins_new_model.log', '15')
plot_ppo_ddpg('replay_ppo_15mins_new_model.log', 'replay_ddpg_15mins_new_model.log', '15')
plot_ppo_ddpg('replay_ppo_1mins_new_model.log', 'replay_ddpg_1mins_new_model.log', '1')

plot_training('training_ddpg_1min.log', 'DDPG', '1')
plot_training('training_ddpg_15mins.log', 'DDPG', '15')
# plot_training('training_ddpg_30mins_new_model.log', 'DDPG', '30')
plot_training('training_ddpg_30mins_old_model.log', 'DDPG','30')
plot_training('training_ppo_1min.log', 'PPO', '1')
plot_training('training_ppo_15mins.log', 'PPO', '15')
plot_training('training_ppo_30mins.log', 'PPO', '30')
