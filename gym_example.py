import gym
from gym import spaces
import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

import pybullet_data

import datetime

import time
import numpy as np
import random


class DiscreteRandomPolicy(object):
    """
    input: n x observations
    output: n x actions
    """
    def __init__(self):
       self.int_range = [1,8]

    def sample_action(self, obs):
       return random.randint(*self.int_range)


class ContinuousDownwardBiasPolicy(object):
  """Policy which takes continuous actions, and is biased to move down.
  """

  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.

    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(5,))

  def sample_action(self, obs, explore_prob=0.1):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da, close = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da, 0]

def get_low_dim_observations(env):
    # get robot states
    robot_states = env._kuka.getObservation()
    # get all object states
    obj_pos_list = []  # position of objects
    obj_orn_list = []  # orientation of objects
    for obj_id in env._objectUids:
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        obj_pos_list.append(obj_pos)
        obj_orn_list.append(obj_orn)
    return [robot_states, obj_pos_list, obj_orn_list]

if __name__ == '__main__':
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=False, removeHeightHack=True)  #KukaDiverseObjectEnv
    print(env.action_space.shape)
    # a continous action policy
    policy = ContinuousDownwardBiasPolicy()

    # a discrete action policy
    # policy = DiscreteRandomPolicy()

    while True:
        obs, done = env.reset(), False  # obs is a RGB image
        low_dim_obs = get_low_dim_observations(env)  # get low-dim obs

        print("===================================")
        print("obs")
        print(obs)

        print('low dim obs')
        print(low_dim_obs)
        
        episode_rew = 0
        while not done:
            env.render(mode='human')
            act = policy.sample_action(obs)
            print("Action")
            print(act)
            obs, rew, done, _ = env.step(act)
            episode_rew += rew
        print("Episode reward", episode_rew)
