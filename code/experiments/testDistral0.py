import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import play_game
sys.path.append('../distral_2col0')
import trainingDistral2col0

models, policy, episode_rewards, episode_durations = trainingDistral2col0.trainD()
