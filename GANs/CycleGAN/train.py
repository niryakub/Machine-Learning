import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import os


def train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loss_eval_cycle = 0
    loss_eval_disc = 0
    loss_eval_gen = 0
    loss_eval_ident = 0

    with tqdm.tqdm(total=len(loader), file=sys.stdout) as pbar:
        for idx, (zebra, horse) in enumerate(loader):
            zebra = zebra.to(config.DEVICE)
            horse = horse.to(config.DEVICE)

            # Train Discriminators H and Z:
            with torch.cuda.amp.autocast(): # torch.cuda.amp provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and other operations use torch.float16 (half). Some ops, like linear layers and convolutions, are much faster in float16. Other ops, like reductions, often require the dynamic range of float32. Mixed precision tries to match each op to its appropriate datatype.

                # Training over the following path: zebra -> fake_horse (thus over gen_H, disc_H)
                fake_horse = gen_H(zebra)
                D_H_real = disc_H(horse) # apply disc' over real image of a horse
                D_H_fake = disc_H(fake_horse.detach()) # apply disc' over fake image of a horse.  detach() returns a new Tensor, detached from the current graph - we don't want to effect the Generator's weights when backpropagating later on.

                # Calculate GANs loss, according to paper where MSE will replace the neg-log-likelihood for stabilizing the training procedure,
                D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real)) # MSE between disc' output and matrices of ones which indicating "real patches" since we run the disc' over a real image
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake)) # same logics ^^
                D_H_loss = D_H_real_loss + D_H_fake_loss


                # Training over the following path: horse -> fake_zebra
                fake_zebra = gen_Z(horse)
                D_Z_real = disc_Z(zebra)  # apply disc' over real image of a horse
                D_Z_fake = disc_Z(fake_zebra.detach())  # apply disc' over fake image of a zebra.  detach() returns a new Tensor, detached from the current graph.

                # Calculate GANs loss, according to paper where MSE will replace the neg-log-likelihood for stabilizing the training procedure,
                D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))  # MSE between disc' output and matrices of ones which indicating "real patches" since we run the disc' over a real image
                D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))  # same logics ^^
                D_Z_loss = D_Z_real_loss + D_Z_fake_loss

                # Plugging it together:
                D_loss = (D_H_loss + D_Z_loss)/2 # divide by 2 according to paper

            # Backpropagate over D_loss:
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward() # applying grad-scaling
            d_scaler.step(opt_disc) # only updating discriminators' weights
            d_scaler.update()

            # Train Generators H & Z:
            with torch.cuda.amp.autocast():
                # Propagate through discriminators with the fake images, per discriminator:
                D_H_fake = disc_H(fake_horse)
                D_Z_fake = disc_Z(fake_zebra)

                # Calculate MSE losses accordingly for generators:
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake)) # MSE between disc' output (of fake-img-input!), and matrices of 1's, since we want to FOOL the disc' (training generators here)
                loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake)) # same logics ^^


                # Cycle losses:
                cycle_zebra = gen_Z(fake_horse) # grab generated fake horse (which was made out of real zebra) and generate a zebra
                cycle_horse = gen_H(fake_zebra)
                cycle_zebra_loss = l1(zebra, cycle_zebra)
                cycle_horse_loss = l1(horse, cycle_horse)

                # Identity losses (Although the paper for this particular dataset, isn't using these specific losses, hence defined config.LAMBDA_IDENTITY=0):
                identity_zebra = gen_Z(zebra) # generate zebra out of zebra (should result in identical output..)
                identity_horse = gen_H(horse)
                identity_zebra_loss = l1(zebra, identity_zebra)
                identity_horse_loss = l1(horse, identity_horse)

                # Plug it all together:
                G_loss = loss_G_Z + loss_G_H + cycle_zebra_loss*config.LAMBDA_CYCLE + cycle_horse_loss*config.LAMBDA_CYCLE + identity_horse_loss*config.LAMBDA_IDENTITY + identity_zebra_loss*config.LAMBDA_IDENTITY

            # Backpropagate:
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()  # applying grad-scaling
            g_scaler.step(opt_gen) # only updating generators' weights
            g_scaler.update()

            # Save some of the generated horses & zebras:
            if idx % 200 == 0:
                save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png") # de-normalize and save img
                save_image(fake_zebra*0.5+0.5, f"saved_images/horse_{idx}.png") # de-normalize and save img


            # Update pbar:
            loss_eval_cycle += cycle_zebra_loss.item() + cycle_horse_loss.item()
            loss_eval_disc += D_loss.item()
            loss_eval_gen += G_loss.item()
            loss_eval_ident += identity_zebra_loss.item() + cycle_horse_loss.item()
            pbar.update();
            pbar.set_description(f'cycle loss={loss_eval_cycle/(idx+1):.3f} | disc loss={loss_eval_disc/(idx+1):.3f} | gen loss={loss_eval_gen/(idx+1):.3f} | ident loss={loss_eval_ident/(idx+1):.3f}')

        pbar.set_description(f'cycle loss={loss_eval_cycle/len(loader):.3f} | disc loss={loss_eval_disc/len(loader):.3f} | gen loss={loss_eval_gen/len(loader):.3f} | ident loss={loss_eval_ident/len(loader):.3f}')
        pbar.update();



def main():
    # Initialize generators & discriminators:
    disc_H = Discriminator(in_channels=3).to(config.DEVICE) # to discriminate if image is a real/fake horse
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE) # to discriminate if image is a real/fake zebra
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # to generate a zebra out of horse
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # to generate a horse out of zebra

    # Initialize optimizers, with the gen's parameters "concatenated", and like wise with disc's. lr and betas according to paper.
    opt_disc = optim.Adam( list(disc_H.parameters()) + list(disc_Z.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam( list(gen_Z.parameters()) + list(gen_H.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)

    # Initialize losses:
    L1 = nn.L1Loss() # for the cycle-loss
    mse = nn.MSELoss() # for the gans-loss

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECK_POINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECK_POINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_DISC_H, disc_H, opt_disc, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, config.LEARNING_RATE,)

    # Creating datasets & dataloaders:
    dataset = HorseZebraDataset(root_horse=config.TRAIN_DIR+"/trainA", root_zebra=config.TRAIN_DIR+"/trainB", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    # GradScaler(): If we run on float32/64, it's not really required...
    # If the forward pass for a particular op has float16 inputs, the backward pass for that op will produce float16 gradients.
    # Gradient values with small magnitudes may not be representable in float16.
    # These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost.
    # so GradScaler prevents above issue by multiplying network's loss by a scale factor and invokes a backward pass on the scaled loss and gradients flowing backward through the network are then scaled by the same factor.
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Train for #epochs:
    for epoch in range(config.NUM_EPOCHS):
        train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECK_POINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECK_POINT_GEN_Z)
            save_checkpoint(disc_H, opt_gen, filename=config.CHECK_POINT_DISC_H)
            save_checkpoint(disc_Z, opt_gen, filename=config.CHECK_POINT_DISC_Z)


if __name__ == "__main__":
    main()