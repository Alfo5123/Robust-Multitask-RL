import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, Tensor
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

env = GridworldEnv(1) # Number of plan
# plt.ion()

num_actions = env.action_space.n
model = DQN(num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.RMSprop(model.parameters(), )

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

memory = ReplayMemory(10000)

last_sync = 0

def get_screen():
    # TODO: may have some bugs
    screen = env.current_grid_map
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).unsqueeze(0).type(Tensor)

episode_durations = []
mean_durations = []
episode_rewards = []
mean_rewards = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    # durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, label="durations")
    # Take 100 episode averages and plot them too
    # if len(episode_durations) >= 100:
        # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(99), means))
    mean_durations.append(np.mean(np.array(episode_durations)[::-1][:100]))
    plt.plot(mean_durations, label="means")
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())

def plot_rewards():
    plt.figure(3)
    plt.clf()
    plt.title('DQN - Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label="Reward per Episode")
    mean_rewards.append(np.mean(np.array(episode_rewards)[::-1][:100]))
    plt.plot(mean_rewards, label="Mean reward")
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())

def plot_state(state):
    if state is not None:
        plt.figure(1)
        plt.clf()
        plt.imshow(state.cpu().squeeze(0).squeeze(0).numpy(),
                       interpolation='none')
        plt.draw()
        plt.pause(0.000001)


def trainDQN(file_name="DQN"):

    ### DQN training routine. Retuns rewards and durations logs.

    ## Plot environment screen
    #env.reset()
    #plt.figure()
    #plt.imshow(get_screen().cpu().squeeze(0).squeeze(0).numpy(),
    #           interpolation='none')
    #plt.draw()
    # plt.pause(0.0001)

    steps_done = 0
    num_episodes = 500 # TODO: 10 is too small number!
    max_num_of_steps = 1000
    for i_episode in range(num_episodes):
        print("Cur episode:", i_episode, "steps done:", steps_done,
                "exploration factor:", EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY))
        # Initialize the environment and state
        env.reset()
        # last_screen = env.current_grid_map
        current_screen = get_screen()
        state = current_screen # - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, model, num_actions,
                                    EPS_START, EPS_END, EPS_DECAY, steps_done)
            steps_done += 1
            _, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            # plot_state(state)
            # env.render()

            # Perform one step of the optimization (on the target network)
            optimize_model(model, optimizer, memory, BATCH_SIZE, GAMMA)
            if done or t + 1 >= max_num_of_steps:
                episode_durations.append(t + 1)
                episode_rewards.append(env.episode_total_reward)
                # plot_durations()
                # plot_rewards()
                break

    print('Complete')
    env.render(close=True)
    env.close()

    plt.ioff()
    plt.show()

    ## Store Results}

    np.save(file_name + '-Rewards', episode_rewards)
    np.save(file_name + '-Durations', episode_durations)
    
    return episode_rewards, episode_durations
