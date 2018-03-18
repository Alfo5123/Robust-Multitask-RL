import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20

dqn7_rewards = np.load('TEST_DQN/env7-dqn-rewards.npy')[:200]
dqn8_rewards = np.load('TEST_DQN/env8-dqn-rewards.npy')[:200]

distral7_rewards = np.load('TEST_DISTRAL/Distral_2col-78-rewards.npy')[0][:200]
distral8_rewards = np.load('TEST_DISTRAL/Distral_2col-78-rewards.npy')[1][:200]

dqn7_smooth = pd.Series(dqn7_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dqn8_smooth = pd.Series(dqn8_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()

dist7_smooth = pd.Series(distral7_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dist8_smooth = pd.Series(distral8_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()

plt.figure(figsize=(20,10))
plt.title('Benchmark Training Results DQN vs Distral', fontsize='20')
plt.xlabel('Episodes ', fontsize='16')
plt.ylabel('Reward', fontsize='16')


plt.plot(dqn7_smooth, label="Env 7 - DQN")
plt.plot(dqn8_smooth, label="Env 8 - DQN")

plt.plot(dist7_smooth, label="Env 7 - DISTRAL")
plt.plot(dist8_smooth, label="Env 8 - DISTRAL")

plt.legend(loc='best', fontsize='20')
plt.savefig('Benchmark-dqn-vs-distral78-reward.eps', format='eps', dpi=1000)
# plt.show()
# plt.close()
