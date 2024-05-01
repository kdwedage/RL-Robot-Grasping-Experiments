import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
argparser.add_argument('--latent_dim', type=int, default=10, help='Dimensionality of the latent space')
args = argparser.parse_args()

torch.manual_seed(args.seed)

# Set the random seed for NumPy
np.random.seed(args.seed)

# Define the VAE model class (same as before)...
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

# Initialize the VAE model
vae = VAE(latent_dim=args.latent_dim)

# Load the trained model state from the saved file
vae.load_state_dict(torch.load('vae_10.pth'))  # Replace 'vae.pth' with the path to your saved model file

# Set the model to evaluation mode
vae.eval()

# Define a function to generate an example image from random mean and logvar values
def generate_image_from_random_mean_logvar(vae, mean, logvar):
    with torch.no_grad():
        # Generate random latent variables from mean and logvar
        z = vae.reparameterize(mean, logvar)

        # Decode the latent variables to generate an image
        reconstructed_image = vae.decoder(z.unsqueeze(0))  # Unsqueeze to add batch dimension
        reconstructed_image = reconstructed_image.view(3, 48, 48)  # Reshape to image dimensions

    return reconstructed_image

# Example random mean and logvar values (replace with your own values)
mean = torch.randn(args.latent_dim)  # Assuming latent dimension is 20
logvar = torch.randn(args.latent_dim)  # Assuming latent dimension is 20

# Generate an example image from random mean and logvar values
example_image = generate_image_from_random_mean_logvar(vae, mean, logvar)
example_image = example_image.permute(1, 2, 0).cpu().numpy()  # Reshape to (48, 48, 3) and convert to NumPy array
print('Shape: ' + str(example_image.shape))
# Convert the tensor to numpy array and plot the image
plt.imshow(example_image)
plt.axis('off')
plt.show()
plt.savefig(f'example_image_{args.seed}.png')