import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

smoothing_window = 20

dqn1_rewards = np.load('TEST_A3C/env1-a3c-rewards.npy')
dqn4_rewards = np.load('TEST_A3C/env4-a3c-rewards.npy')
dqn5_rewards = np.load('TEST_A3C/env5-a3c-rewards.npy')
dqn7_rewards = np.load('TEST_A3C/env7-a3c-rewards.npy')
dqn8_rewards = np.load('TEST_A3C/env8-a3c-rewards.npy')

dqn1_smooth = pd.Series(dqn1_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dqn4_smooth = pd.Series(dqn4_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dqn5_smooth = pd.Series(dqn5_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dqn7_smooth = pd.Series(dqn7_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()
dqn8_smooth = pd.Series(dqn8_rewards).rolling(smoothing_window,min_periods=smoothing_window).mean()

plt.figure(figsize=(20,10))
plt.title('Benchmark Training Results A3C', fontsize='18')
plt.xlabel('Episode Reward (Smoothed)', fontsize='14')
plt.ylabel('Rewards', fontsize='14')
plt.plot(dqn1_smooth, label="Env 1")
plt.plot(dqn4_smooth, label="Env 4")
plt.plot(dqn5_smooth, label="Env 5")
plt.plot(dqn7_smooth, label="Env 7")
plt.plot(dqn8_smooth, label="Env 8")
plt.legend(loc='best', fontsize='20')
plt.savefig('Benchmark-a3c-reward.eps', format='eps', dpi=1000)
# plt.show()
# plt.close()
