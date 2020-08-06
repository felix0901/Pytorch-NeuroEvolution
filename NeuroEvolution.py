
import copy
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import pickle
import time
from torch import FloatTensor
import random
import numpy as np
import torch
from operator import add

class NeuroEvolution:

    def __init__(
        self, 
        weights, 
        reward_func,
        population_size=50,
        sigma=0.01,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None,
        candidate_num = 10,
        cand_test_time = 10,
        method = 2,
        seeded_env=-1
    ):
        np.random.seed(int(time.time()))
        self.cand_test_times = cand_test_time
        self.weights = weights
        self.reward_function = reward_func
        self.candidate_num = max(1, candidate_num)
        self.POPULATION_SIZE = max(population_size, self.candidate_num)
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.decay = decay
        self.sigma_decay = sigma_decay
        if cuda and torch.cuda.is_available():
            self.pool = ThreadPool(threadcount)
        else:
            self.pool = Pool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path
        self.method = method
        self.seeded_env=seeded_env

    # def reward_func_wrapper(self):


    def mutate(self, parent_list, sigma):
        child_list = []
        for parent in parent_list[0]:
            child = parent + sigma *  torch.from_numpy(np.random.normal(0,0.2,parent.shape)).type(torch.FloatTensor).to(self.device)
            # child = (torch.from_numpy(np.random.randint(0,2,parent.shape))).type(torch.DoubleTensor).to(self.device)
            child_list.append(child)
        return child_list

    def run(self, iterations, print_step=10):
        for iteration in range(iterations):
            n_pop = []
            for i in range(self.POPULATION_SIZE):
                if iteration == 0:
                    x = []
                    for param in self.weights:
                        x.append(torch.from_numpy(np.random.randn(*param.data.size())).type(torch.FloatTensor).to(self.device))
                        # x.append((torch.from_numpy(np.random.randint(0, 2,param.data.size()))).to(self.device))
                    n_pop.append([x, 0, i])
                else:
                    # p_id = random.randint(0, self.POPULATION_SIZE-1)
                    p_id = i
                    new_p = self.mutate(pop[p_id], self.SIGMA)
                    n_pop.append([copy.deepcopy(new_p), 0, i])
            rewards = self.pool.map(
                self.reward_function,
                [p[0] for p in n_pop]
            )
            for i, _ in enumerate(n_pop):
                n_pop[i][1] = rewards[i]
            n_pop.sort(key=lambda p: p[1], reverse=True)
            for i in range(self.candidate_num):
                n_pop[i][2] = i

            if self.seeded_env >= 0:
                if iteration==0:
                    elite=n_pop[0]
                else:
                    elite = max([n_pop[0], prev_elite], key=lambda p: p[1])
            else:
                if iteration==0:
                    elite_c = n_pop[:self.candidate_num]
                else:
                    elite_c = n_pop[:self.candidate_num-1] + [prev_elite]

                rewards_list = np.zeros((10,))
                for _ in range(self.cand_test_times):
                    rewards = self.pool.map(
                        self.reward_function,
                        [p[0] for p in elite_c]
                    )

                    rewards_list += np.array(rewards)
                rewards_list/=self.cand_test_times
                for i, _ in enumerate(elite_c):
                    elite_c[i][1] = rewards_list[i]
                elite = max(elite_c, key=lambda p: p[1])
            if self.method==1:
                n_pop[elite[2]] = elite
            else:
                if iteration != 0:
                    n_pop[-1] = prev_elite
            pop = n_pop
            prev_elite = elite
            prev_elite[2] = -1




            test_reward = self.reward_function(
                elite[0], render=self.render_test
            )
            if (iteration+1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration+1, test_reward))
                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))
                
            if self.reward_goal and self.consecutive_goal_stopping:
                if test_reward >= self.reward_goal:
                    self.consecutive_goal_count += 1
                else:
                    self.consecutive_goal_count = 0

                if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                    return elite[0]

        return elite[0]