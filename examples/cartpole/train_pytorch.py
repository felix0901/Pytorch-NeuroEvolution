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
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, default="out.plt", help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(True),
    nn.Linear(4, 2),
    # nn.Softmax(dim=1)
)


model= model.to(device)

def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("CartPole-v0")
    env.seed(0)
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        batch = torch.from_numpy(ob[np.newaxis,...]).float().to(device)
        prediction = cloned_model(batch)
        action = prediction.data.cpu().clone().numpy().argmax()
        ob, reward, done, _ = env.step(action)

        total_reward += reward

    env.close()
    return total_reward

partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

ne = NeuroEvolution(
    mother_parameters, partial_func, population_size=15, sigma=0.1,
    learning_rate=0.001, threadcount=50, cuda=args.cuda, reward_goal=200,
    consecutive_goal_stopping=10, seeded_env=-1
)


start = time.time()
final_weights = ne.run(200)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

weights = pickle.load(open(os.path.abspath(args.weights_path), 'rb'))
reward = partial_func(weights, render=True)
print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")