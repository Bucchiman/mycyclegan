import sys
from pathlib import Path
from time import time
import torch
from torchvision.utils import make_grid, save_image
from torch import nn
from datetime import datetime, timedelta
from kornia.geometry import transform
from utils import save_model


def sample_images(output_path,
                  dataset_name,
                  epoch,
                  G_AB,
                  G_BA,
                  valid_dataloader,
                  device):
    """Saves a generated sample from the test set"""
    save_img_path = str(Path(output_path).joinpath("images", dataset_name))
    Path(save_img_path).mkdir(parents=True, exist_ok=True)
    imgs = next(iter(valid_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs["A"].to(device)
    fake_B = G_AB(real_A)
    real_B = imgs["B"].to(device)
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid,
               str(Path(save_img_path).joinpath(".".join((str(epoch).zfill(4), "png")))),
               normalize=False)


def train(output_path,
          dataset_name,
          epochs,
          D_A,
          D_B,
          G_AB,
          G_BA,
          train_dataloader,
          optimizer_G,
          optimizer_D_A,
          optimizer_D_B,
          criterion_GAN,
          criterion_identity,
          criterion_cycle,
          checkpoint_epoch,
          lambda_id,
          lambda_cyc,
          fake_A_buffer,
          fake_B_buffer,
          device,
          lr_scheduler_G,
          lr_scheduler_D_A,
          lr_scheduler_D_B,
          input_dict,
          output_shape,
          trans_flag,
          discriminator_img_shape,
          sample_interval,
          checkpoint_interval):
    prev_time = time()
    transform_D = nn.Sequential(transform.Resize(discriminator_img_shape))
    transform_D = transform_D.to(device)
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    G_AB.train()
    G_BA.train()

    for epoch in range(checkpoint_epoch, epochs):
        for i, batch in enumerate(train_dataloader):
            # Set model input
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *output_shape)).to(device)
            fake = torch.zeros((real_A.size(0), *output_shape)).to(device)

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            if trans_flag:
                loss_GAN_AB = criterion_GAN(D_B(transform_D(fake_B)), valid)
                loss_GAN_BA = criterion_GAN(D_A(transform_D(fake_A)), valid)
            else:
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            del loss_GAN_AB
            del loss_GAN_BA

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            del loss_cycle_A
            del loss_cycle_B

            # Total loss
            loss_G = loss_GAN+lambda_cyc*loss_cycle+lambda_id*loss_identity

            loss_G.backward()
            del loss_G
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            if trans_flag:
                loss_real = criterion_GAN(D_A(transform_D(real_A)), valid)
                loss_fake = criterion_GAN(D_A(transform_D(fake_A_.detach())), fake)
            else:
                loss_real = criterion_GAN(D_A(real_A), valid)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)

            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            if trans_flag:
                loss_real = criterion_GAN(D_B(transform_D(real_B)), valid)
                loss_fake = criterion_GAN(D_B(transform_D(fake_B_.detach())), fake)
            else:
                loss_real = criterion_GAN(D_B(real_B), valid)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)

            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            loss_D = (loss_D_A + loss_D_B) / 2

            del loss_D_A
            del loss_D_B
            optimizer_D_B.step()


            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch*len(train_dataloader)+i
            batches_left = epochs*len(train_dataloader)-batches_done
            time_left = timedelta(seconds=batches_left*(time()-prev_time))
            prev_time = time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] ETA: %s\n"
                % (
                    epoch,
                    epochs,
                    i,
                    len(train_dataloader),
                    time_left,
                )
            )

#            sys.stdout.write(
#                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s\n"
#                % (
#                    epoch,
#                    epochs,
#                    i,
#                    len(train_dataloader),
#                    loss_D.item(),
#                    loss_G.item(),
#                    loss_GAN.item(),
#                    loss_cycle.item(),
#                    loss_identity.item(),
#                    time_left,
#                )
#            )

            # If at sample interval save image
#            if i == len(train_dataloader)-1 and (epoch+1) % sample_interval == 0:
#                sample_images(output_path,
#                              dataset_name,
#                              epoch,
#                              G_AB,
#                              G_BA,
#                              valid_dataloader,
#                              device)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch % checkpoint_interval == 0 or epoch+1 == epochs:
            # Save model checkpoints
            saved_models_path = str(Path(output_path).joinpath("saved_models"))
            save_model(G_AB, "{}/G_AB_{}.pth".format(saved_models_path, str(epoch).zfill(4)))
            save_model(G_BA, "{}/G_BA_{}.pth".format(saved_models_path, str(epoch).zfill(4)))
            save_model(D_A, "{}/D_A_{}.pth".format(saved_models_path, str(epoch).zfill(4)))
            save_model(D_B, "{}/D_B_{}.pth".format(saved_models_path, str(epoch).zfill(4)))
