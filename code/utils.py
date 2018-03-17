import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import torch
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor

def get_screen(env):
    screen = env.current_grid_map
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).unsqueeze(0).type(Tensor)

def play_game(env, agent, max_steps_num=100): #### To fix for dqn0, sql0

    keyFrames = []
    env.reset()
    plt.ion()
    for steps_done in range(0, max_steps_num):
        plt.figure(1)
        state = get_screen(env)
        plt.imshow(state.cpu().squeeze(0).squeeze(0).numpy(),
                   interpolation='none')

        filename = "env4_0" + str(steps_done) + ".png"
        plt.savefig(filename)
        keyFrames.append(filename)

        plt.draw()
        plt.pause(0.1)
        action = agent(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action[0, 0])

        if done:
            state = get_screen(env)
            plt.imshow(state.cpu().squeeze(0).squeeze(0).numpy(),interpolation='none')

            filename = "env4_0" + str(steps_done+1) + ".png"
            plt.savefig(filename)
            keyFrames.append(filename)
            break
    plt.ioff()

    images = [imageio.imread(fn) for fn in keyFrames]
    gifFilename = "env4-solution.gif"
    imageio.mimsave(gifFilename, images, duration=0.3)



def plot_durations(episode_durations, mean_durations):
    plt.figure(2)
    plt.clf()
    # durations_t = torch.FloatTensor(episode_durations)
    plt.title('Durations, training...')
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

def plot_rewards(episode_rewards, mean_rewards):
    plt.figure(3)
    plt.clf()
    plt.title('Rewards, training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label="Reward per Episode")
    mean_rewards.append(np.mean(np.array(episode_rewards)[::-1][:100]))
    plt.plot(mean_rewards, label="Mean reward")
    plt.legend()

    plt.pause(0.00001)  # pause a bit so that plots are updated
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
