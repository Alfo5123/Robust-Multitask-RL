import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
sys.path.append('../sql')
import trainingSQL


trainingSQL.trainSQL(file_name="env1",
                    env=GridworldEnv(1),
                    batch_size=128,
                    gamma=0.8,
                    beta=5,
                    eps_start=0.9,
                    eps_end=0.05,
                    eps_decay=300,
                    is_plot=False,
                    num_episodes=500,
                    max_num_steps_per_episode=1000,
                    learning_rate=0.001,
                    memory_replay_size=10000,)
