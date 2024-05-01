import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--latent_dim', type=int, default=20, help='Dimension of the latent space.')
argparser.add_argument('--beta', type=float, default=1, help='Beta value for the beta-VAE loss.')
argparser.add_argument('--seed', type=int, default=2, help='Random seed.')
args = argparser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set random seed for GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Additional seeds for certain operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

        self.min_val = float('inf')
        self.max_val = float('-inf')

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
        self.min_val = min(self.min_val, torch.min(z).item())
        self.max_val = max(self.max_val, torch.max(z).item())

        # Decode
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x_recon.size(0), 3, 48, 48)
    
        return x_recon, mu, logvar

    def get_min_max(self):
        return self.min_val, self.max_val

# Define your custom dataset
class NumpyArrayDataset(Dataset):
    def __init__(self, numpy_array, transform=None):
        self.data = numpy_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
latent_dim = args.latent_dim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize your VAE model
vae = VAE(latent_dim=latent_dim).to(device)



numpy_data = np.load('observations.npy') 
dataset = NumpyArrayDataset(numpy_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + args.beta * KLD

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    vae.train()
    total_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0 and False:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item() / len(data)))

    #print('====> Epoch: {} Average loss: {:.4f}'.format(
    #      epoch, total_loss / len(dataloader.dataset)))
print('Min value in latent space:', vae.get_min_max()[0])
print('Max value in latent space:', vae.get_min_max()[1])

# Save the trained model
torch.save(vae.state_dict(), f'vae_latent{args.latent_dim}_beta{args.beta}_{vae.get_min_max()[0]}_{vae.get_min_max()[1]}.pth')
