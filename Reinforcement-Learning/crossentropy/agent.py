import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

from collections import namedtuple
from tensorboardX import SummaryWriter
from network import *



class Agent(object):
    def __init__(self, args):
        self.env            = gym.make(args.env)
        
        self.obs_size       = self.env.observation_space.shape[0]
        self.n_actions      = self.env.action_space.n
        self.batch_size     = args.batch_size
        self.net            = Net(self.obs_size, args.hidden_size, self.n_actions)
        self.loss_fn        = nn.CrossEntropyLoss()
        self.optimizer      = optim.Adam(self.net.parameters(), lr= args.lr)
        self.writer         = SummaryWriter(comment = args.env)
        self.percentile     = args.percentile
        self.total_reward   = 0


    def get_batch(self):
        batch = []
        Episode = namedtuple('Episode', field_names=['reward', 'steps'])
        EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
        
        while len(batch) != self.batch_size:
            finish = False
            episode_steps = []
            episode_reward = 0
            obs = self.env.reset()
           
            while not finish:
                obs_v    = torch.FloatTensor([obs])
                actoin_prob = F.softmax(self.net(obs_v), 1)
                actoin_prob = actoin_prob.detach().numpy()[0]
                action = np.random.choice(len(actoin_prob), p = actoin_prob)
                next_obs, reward, is_done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps.append(EpisodeStep(observation = obs, action = action))
                if is_done:
                    finish = is_done
                    batch.append(Episode(reward = episode_reward, steps = episode_steps))
                obs = next_obs
        return batch

    def filter_batch(self):
        batch = self.get_batch()
        reward = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(reward, self.percentile)
        reward_mean = float(np.mean(reward))
        train_obs = []
        train_act = []

        for sample in batch:
            if sample.reward >= reward_bound:
                train_obs.extend(map(lambda s: s.observation, sample.steps))
                train_act.extend(map(lambda s: s.action, sample.steps))
        
        train_obs = torch.FloatTensor(train_obs)
        train_act = torch.LongTensor(train_act)

        return train_obs, train_act, reward_bound, reward_mean
    

    def solve(self):
        cnt = 0
        finish = False
        while not finish:
            train_obs, train_act, reward_bound, reward_mean = self.filter_batch()
            action = self.net(train_obs)
            loss   = self.loss_fn(action, train_act)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('Round: %d || Loss: %.4f || Reward mean: %.4f || reward_bound = %.4f' % (cnt, loss.item(), reward_mean, reward_bound))
            self.writer.add_scalar("loss", loss.item(), cnt)
            self.writer.add_scalar("reward_bound", reward_bound, cnt)
            self.writer.add_scalar("reward_mean", reward_mean, cnt)
            cnt += 1
            if reward_mean > 199:
                print('Solve!')
                finish = True
        self.writer.close()