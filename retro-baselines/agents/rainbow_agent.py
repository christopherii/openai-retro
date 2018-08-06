#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

import random

import gym
import numpy as np

import gym_remote.client as grc
from baselines.common.atari_wrappers import WarpFrame

import time

from sonic_util import AllowBacktracking, SonicDiscretizer


EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6) # changed for 1e6

def main():
    env = grc.RemoteEnv('tmp/sock')

    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    all_time_steps = 0
    avg_reward = 0
    count = 0
    while True:
      if new_ep:
          if (solutions and
                  random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
              solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
              best_pair = solutions[-1]
              new_rew = exploit(env, best_pair[1])
              avg_reward += new_rew
              count += 1
              #print(count)
              #all_time_steps += env.total_steps_ever
              best_pair[0].append(new_rew)
              print('replayed best with reward %f' % new_rew)
              continue
          else:
              env.reset()
              new_ep = False
      rew, new_ep = move(env, 100)
      if not new_ep and rew <= 0:
          print('backtracking due to negative reward: %f' % rew)
          _, new_ep = move(env, 70, left=True)
      if new_ep:
          solutions.append(([max(env.reward_history)], env.best_sequence()))

      #print(count)
      if count > 20: #100 for running, 7 for testing
        avg_reward = avg_reward / count
        #print(avg_reward)
        if avg_reward < 2600 or avg_reward > 5000: # might want to lower latter to 4500-4700
          break
    """Run DQN until the environment throws an exception."""
    #print(env)
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    env = WarpFrame(env)
    env = AllowBacktracking(env)
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    #print('here 5')
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop. ## Changed from 2000000
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done

def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            _, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
