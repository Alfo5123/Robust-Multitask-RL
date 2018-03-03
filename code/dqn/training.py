import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
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
EPS_DECAY = 200

env = GridworldEnv(1) # Number of plan
plt.ion()

num_actions = env.action_space.n
model = DQN(num_actions)

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

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).squeeze(0).numpy(),
           interpolation='none')
plt.draw()
# plt.pause(0.0001)

num_episodes = 10 # TODO: this is too small number!
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    # last_screen = env.current_grid_map
    current_screen = get_screen()
    state = current_screen # - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state, model, num_actions,
                                EPS_START, EPS_END, EPS_DECAY)
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

        # Perform one step of the optimization (on the target network)
        optimize_model(model, memory, BATCH_SIZE, GAMMA)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
# env.render(close=True)
env.close()
plt.ioff()
plt.show()
