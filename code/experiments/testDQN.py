import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import play_game
sys.path.append('../dqn')
import trainingDQN



# trainingDQN.trainDQN(file_name="env1",
#                     env=GridworldEnv(1),
#                     batch_size=128,
#                     gamma=0.999,
#                     eps_start=0.9,
#                     eps_end=0.05,
#                     eps_decay=1000,
#                     is_plot=True,
#                     num_episodes=500,
#                     max_num_steps_per_episode=1000,
#                     learning_rate=0.0001,
#                     memory_replay_size=10000,
#                 )

agent, _, _ = trainingDQN.trainDQN(file_name="env1",
                    env=GridworldEnv(1),
                    batch_size=128,
                    gamma=0.999,
                    eps_start=0.9,
                    eps_end=0.05,
                    eps_decay=1000,
                    is_plot=False,
                    num_episodes=500,
                    max_num_steps_per_episode=1000,
                    learning_rate=0.0001,
                    memory_replay_size=10000,
                )

play_game(GridworldEnv(1), agent)
