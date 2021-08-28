import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

# GLOBALS:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4 # according to paper
BATCH_SIZE = 64 # according to paper (m=64)
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5 # according to paper, this tells how many iterations we perform over discriminator, per a single iteration over generator
LAMBDA_GP = 10 # gradient penalty lambda according to paper


# Data augmentations:
transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# Initialize dataset & dataloader:
dataset = datasets.CelebA(root="celeb_dataset", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize generator and discriminator(aka critic) & its' weights:
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(DEVICE)
initialize_weights(generator)
initialize_weights(discriminator)

# Initialize optimizers:
opt_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)) # RMSprop is now changed to Adam which is used in the paper.
opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

generator.train()
discriminator.train()

# Perform training:
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(DEVICE)


        # Train Discriminator according to wgan-formula:
        for _ in range(CRITIC_ITERATIONS):
            # Generate fake images out of random latent space z:
            z_vector = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)
            fake = generator(z_vector)

            # Propagate thourgh disc' & calculate loss:
            disc_real = discriminator(real).reshape(-1)
            disc_fake = discriminator(fake).reshape(-1)
            gp = gradient_penalty(discriminator, real, fake, device=DEVICE)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp # the 1st '-' is because we'd like to optimize that expression, yet torch-optimizer minimize by default.

            # Backprop' & update disc' weights:
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # CLIPPING IS REMOVED.


        # Train Generator ( according to -E[critic(generator(z))] ):
        output = discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward()
        opt_generator.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            generator.eval()
            discriminator.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = generator(z_vector)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            generator.train()
            discriminator.train()

