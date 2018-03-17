import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import play_game
sys.path.append('../distral_2col')
import trainingDistral2col

models, policy, episode_rewards, episode_durations = trainingDistral2col.trainD()