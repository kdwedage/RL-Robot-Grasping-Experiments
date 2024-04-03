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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# Define the VAE model with ResNet backbone
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        # ResNet as encoder backbone
        resnet = models.resnet18(pretrained=True)
        self.encoder_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (avgpool and fc)

        # Additional layers for encoder
        self.encoder_mean = nn.Linear(2048, latent_dim)
        self.encoder_logvar = nn.Linear(2048, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 48 * 48),
            nn.Sigmoid()  # To ensure values are between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder_backbone(x)
        x = x.view(x.size(0), -1)  # Flatten the output of ResNet

        # Get mean and logvar
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x_recon.size(0), 3, 48, 48)
    
        return x_recon, mu, logvar

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
])

class CustomKukaDiverseObjectEnv(KukaDiverseObjectEnv):
    def __init__(self, *args, **kwargs):
        self.save_observations = kwargs['save_observations']
        del kwargs['save_observations']
        self.use_vae = kwargs['vae']
        del kwargs['vae']
        self.latent_dim = kwargs['latent_dim']
        del kwargs['latent_dim']
        self.observations = []
 
        if self.use_vae:
            vae = VAE(latent_dim=self.latent_dim)
            vae.load_state_dict(torch.load(self.use_vae))  # Load the saved VAE model
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
            self.vae = vae
   
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

        if self.use_vae:
            np_img_arr = np_img_arr[:, :, :3]
            image_tensor = transform(np_img_arr).unsqueeze(0)
            with torch.no_grad():
                x_recon, _, _ = self.vae(image_tensor)
        return x_recon.squeeze().permute(1,2,0)

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
    parser.add_argument('--path_to_vae', type=str, default=None, help='Path to the VAE model.')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension of the VAE.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    # create the env
    env = CustomKukaDiverseObjectEnv(renders=args.render, isDiscrete=False, 
                                     removeHeightHack=True, maxSteps=args.max_steps, 
                                     save_observations=args.save_observations, 
                                     vae=args.path_to_vae, latent_dim=args.latent_dim)

    # create a PPO
    if args.path_to_vae:
        args.tb_log_name = f"{args.tb_log_name}_{args.latent_dim}"
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
