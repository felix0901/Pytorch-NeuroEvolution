import argparse
import copy
from functools import partial
import logging

import time

from NeuroEvolution import NeuroEvolution

import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, default="out.plt", help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

model = nn.Sequential(
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    # nn.Softmax(1)
)

if cuda:
    model = model.cuda()


def get_reward(weights, model, render=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("LunarLander-v2")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.01)
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch))
        action = prediction.data.cpu().clone().numpy().argmax()
        ob, reward, done, _ = env.step(action)

        total_reward += reward 
    env.close()
    return total_reward

partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())



ne = NeuroEvolution(
    mother_parameters, partial_func, population_size=15, sigma=0.1,
    learning_rate=0.001, threadcount=200, cuda=args.cuda, reward_goal=200,
    consecutive_goal_stopping=10, seeded_env=-1
)
start = time.time()
final_weights = ne.run(4000, print_step=10)
end = time.time() - start

# reward = partial_func(final_weights, render=False)
