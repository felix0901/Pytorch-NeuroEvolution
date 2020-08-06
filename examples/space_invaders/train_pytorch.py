import argparse
import copy
from functools import partial
import logging
import os
import pickle
import sys
import time

from NeuroEvolution import NeuroEvolution

import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, default="out.plt", help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)
args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

num_features = 16
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

model = nn.Sequential(
    nn.Linear(128, 200),
    nn.ReLU(),
    nn.Linear(200, 500),
    nn.ReLU(),
    nn.Linear(500, 6),
    nn.Softmax(1)
)


if cuda:
    model = model.cuda()

env = gym.make("SpaceInvaders-ram-v0")

def get_reward(weights, model, render=False):
    global env

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.005)
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch, volatile=True))
        action = prediction.data.cpu().numpy().argmax()
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
final_weights = ne.run(10000, print_step=10)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))
print("Total time: {}", end)
# reward = partial_func(final_weights, render=True)
