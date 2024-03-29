import os
import numpy as np
# Patch and register pybullet envs
import rl_zoo3.gym_patches
import pybullet_envs

from stable_baselines3 import PPO
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from stable_baselines3.common.evaluation import evaluate_policy
import pybullet as p
import argparse


class CustomKukaDiverseObjectEnv(KukaDiverseObjectEnv):
    def __init__(self, *args, **kwargs):
        self.save_observations = kwargs['save_observations']
        del kwargs['save_observations']
        self.observations = []
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
        if self.save_observations:
            self.observations.append(np_img_arr[:, :, :3])
        return np_img_arr[:, :, :3]

    def save_observation(self, filename='observations.npy'):
        np.save(filename, np.stack(self.observations, axis=0))

def parse_args():
    parser = argparse.ArgumentParser(description='Run PPO training on KukaDiverseObjectEnv.')
    parser.add_argument('--train', action='store_true', help='Whether to train the agent or not.')
    parser.add_argument('--evaluate', action='store_true', help='Whether to evaluate the agent or not.')
    parser.add_argument('--load_from_path', type=str, default=None, help='Path to load the model from.')
    parser.add_argument('--log_dir', type=str, default="/home/kevin/Documents/720/", help='Directory to save the model to.')
    parser.add_argument('--max_steps', type=int, default=20, help='Max number of steps to run the simulation for.')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total number of timesteps to train for.')
    parser.add_argument('--num_eval_episodes', type=int, default=100, help='Number of episodes to evaluate the agent for.')
    parser.add_argument('--tb_log_name', type=str, default="PPO", help='Name of the tensorboard log.')
    parser.add_argument('--seed', type=int, default=2, help='Seed for the random number generator.')
    parser.add_argument('--render', action='store_true', help='Whether to render the environment or not.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for the PPO algorithm.')
    parser.add_argument('--learning_rate', type=float, default=1.e-4, help='Learning rate for the PPO algorithm.')
    parser.add_argument('--save_observations', action='store_true', help='Whether to save observations for training VAE.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    # create the env
    env = CustomKukaDiverseObjectEnv(renders=args.render, isDiscrete=False, removeHeightHack=True, maxSteps=args.max_steps, save_observations=args.save_observations)

    # create a PPO, SAC, TD3
    model = PPO('MlpPolicy', env, verbose=1, seed=args.seed, tensorboard_log=args.log_dir, batch_size=args.batch_size, learning_rate=args.learning_rate)
    save_path = args.log_dir + "ppo_kuka_grasp"

    if args.train:
        model.learn(total_timesteps=100000, progress_bar=True, tb_log_name=args.tb_log_name)

        # save the agent
        model.save(save_path)

    if args.evaluate:
        # load the agent
        if args.load_from_path is None:
            print(f'Loading model from {save_path}')
            model = PPO.load(save_path)
        else:
            print(f'Loading model from {args.load_from_path}')
            model = PPO.load(args.load_from_path)

        # evaluate the agent
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=args.num_eval_episodes)
        print(f"Mean reward: {mean_reward}") # 0.42, for 3:40:42 hours training.

    if args.save_observations:
        env.save_observation()

    # close the env
    env.close()

if __name__ == '__main__':
    main()
