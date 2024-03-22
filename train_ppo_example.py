import os
import numpy as np
# Patch and register pybullet envs
import rl_zoo3.gym_patches
import pybullet_envs

from stable_baselines3 import PPO
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from stable_baselines3.common.evaluation import evaluate_policy
import pybullet as p


class CustomKukaDiverseObjectEnv(KukaDiverseObjectEnv):
    def __init__(self, *args, **kwargs):
        super(CustomKukaDiverseObjectEnv, self).__init__(*args, **kwargs)

    def _get_observation(self):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        return np_img_arr[:, :, :3]


def main(train=True):
    # create the env
    env = CustomKukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=True, maxSteps=20)  #KukaDiverseObjectEnv

    # create a PPO, SAC, TD3
    model = PPO('MlpPolicy', env, verbose=1, seed=2)
    log_dir = "/home/kevin/Documents/720/"

    if train:
        model.learn(total_timesteps=100000, progress_bar=True)

        # save the agent
        model.save(log_dir + "ppo_kuka_grasp")

    else:
        # load the agent
        print('loading model')
        log_dir = "/home/kevin/Documents/720/"
        model = PPO.load(log_dir + "ppo_kuka_grasp")

    # evaluate the agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}") # 0.8, for 3:40:42 hours training.

    # close the env
    env.close()

main(True)