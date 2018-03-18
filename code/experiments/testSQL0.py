import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import play_game
sys.path.append('../sql0')
import trainingSQL0

import gym
agent, _, _ = trainingSQL0.trainSQL0(file_name="env1",
                    env=GridworldEnv(1),
                    batch_size=128,
                    gamma=0.90, ## To Tune
                    beta=5.0, ## To tune
                    eps_start=0.9,
                    eps_end=0.05,
                    eps_decay=10000,
                    is_plot=False,
                    num_episodes=500,
                    max_num_steps_per_episode=10000,
                    learning_rate=0.001,
                    memory_replay_size=10000,
                )

#play_game(GridworldEnv(1), agent)
