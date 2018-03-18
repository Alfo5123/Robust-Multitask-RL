import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, Tensor, optimize_policy, PolicyNetwork
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import plot_rewards, plot_durations, plot_state, get_screen

def trainD(file_name="Distral_1col", list_of_envs=[GridworldEnv(4),
            GridworldEnv(5)], batch_size=128, gamma=0.999, alpha=0.9,
            beta=5, eps_start=0.9, eps_end=0.05, eps_decay=5,
            is_plot=False, num_episodes=200,
            max_num_steps_per_episode=1000, learning_rate=0.001,
            memory_replay_size=10000, memory_policy_size=1000):
    """
    Soft Q-learning training routine. Retuns rewards and durations logs.
    Plot environment screen
    """
    num_actions = list_of_envs[0].action_space.n
    num_envs = len(list_of_envs)
    policy = PolicyNetwork(num_actions)
    models = [DQN(num_actions) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    memories = [ReplayMemory(memory_replay_size, memory_policy_size) for _ in range(0, num_envs)]

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        policy.cuda()
        for model in models:
            model.cuda()

    optimizers = [optim.Adam(model.parameters(), lr=learning_rate)
                    for model in models]
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    episode_durations = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]

    steps_done = np.zeros(num_envs)
    episodes_done = np.zeros(num_envs)
    current_time = np.zeros(num_envs)

    # Initialize environments
    for env in list_of_envs:
        env.reset()

    while np.min(episodes_done) < num_episodes:
        # TODO: add max_num_steps_per_episode

        # Optimization is given by alterating minimization scheme:
        #   1. do the step for each env
        #   2. do one optimization step for each env using "soft-q-learning".
        #   3. do one optimization step for the policy

        for i_env, env in enumerate(list_of_envs):
            # print("Cur episode:", i_episode, "steps done:", steps_done,
            #         "exploration factor:", eps_end + (eps_start - eps_end) * \
            #         math.exp(-1. * steps_done / eps_decay))
        
            # last_screen = env.current_grid_map
            current_screen = get_screen(env)
            state = current_screen # - last_screen
            # Select and perform an action
            action = select_action(state, policy, models[i_env], num_actions,
                                    eps_start, eps_end, eps_decay,
                                    episodes_done[i_env], alpha, beta)
            steps_done[i_env] += 1
            current_time[i_env] += 1
            _, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # Store the transition in memory
            time = Tensor([current_time[i_env]])
            memories[i_env].push(state, action, next_state, reward, time)

            # Perform one step of the optimization (on the target network)
            optimize_model(policy, models[i_env], optimizers[i_env],
                            memories[i_env], batch_size, alpha, beta, gamma)
            if done:
                print("ENV:", i_env, "iter:", episodes_done[i_env],
                    "\treward:", env.episode_total_reward,
                    "\tit:", current_time[i_env], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i_env] / eps_decay))
                env.reset()
                episodes_done[i_env] += 1
                episode_durations[i_env].append(current_time[i_env])
                current_time[i_env] = 0
                episode_rewards[i_env].append(env.episode_total_reward)
                if is_plot:
                    plot_rewards(episode_rewards, i_env)


        optimize_policy(policy, policy_optimizer, memories, batch_size,
                    num_envs, gamma)

    print('Complete')
    env.render(close=True)
    env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    ## Store Results

    np.save(file_name + '-distral-2col-rewards', episode_rewards)
    np.save(file_name + '-distral-2col-durations', episode_durations)

    return models, policy, episode_rewards, episode_durations
