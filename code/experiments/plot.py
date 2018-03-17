import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20

sql_rewards = np.load('env1-sql0-rewards.npy')
dqn_rewards = np.load('env1-dqn0-rewards.npy')
a3c_rewards = np.load('env1-a3c-rewards.npy')

sql_smooth = pd.Series(sql_rewards).rolling(smoothing_window,
                            min_periods=smoothing_window).mean()
dqn_smooth = pd.Series(dqn_rewards).rolling(smoothing_window,
                            min_periods=smoothing_window).mean()
a3c_smooth = pd.Series(a3c_rewards).rolling(smoothing_window,
                            min_periods=smoothing_window).mean()

plt.figure(figsize=(20,10))
plt.title('Benchmark Training Results', fontsize='18')
plt.xlabel('Episode Reward (Smoothed)', fontsize='14')
plt.ylabel('Reward', fontsize='14')
plt.plot(sql_smooth, label="SQL")
plt.plot(dqn_smooth, label="DQN")
plt.plot(a3c_smooth, label="A3C")
plt.legend(loc='best', fontsize='14')
plt.savefig('Benchmark1.eps', format='eps', dpi=1000)
# plt.show()
# plt.close()
