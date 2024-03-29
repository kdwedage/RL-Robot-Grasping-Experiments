import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models

# Define the VAE model with ResNet backbone
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        # ResNet as encoder backbone
        resnet = models.resnet18(pretrained=True)
        self.encoder_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (avgpool and fc)

        # Additional layers for encoder
        self.encoder_mean = nn.Linear(512, latent_dim)
        self.encoder_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 64 * 64),  # Assuming input image size is 64x64 and 3 channels (RGB)
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

        return x_recon, mu, logvar


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
latent_dim = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize your VAE model
vae = VAE(input_shape=(3, 64, 64), latent_dim=latent_dim).to(device)


# Assuming your numpy array is named 'numpy_data'
numpy_data = np.load('path_to_your_numpy_array.npy')  # Replace 'path_to_your_numpy_array.npy' with the path to your numpy array
dataset = NumpyArrayDataset(numpy_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, total_loss / len(dataloader.dataset)))

# Save the trained model
torch.save(vae.state_dict(), 'vae.pth')
