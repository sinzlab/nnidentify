import numpy as np
import torch
from torch import optim
from tqdm import trange


def trainer_fn(gabor_generator, model_neuron,
               epochs=20000, lr=5e-3,
               fixed_norm=10,
               save_rf_every_n_epoch=None,
               optimizer=optim.Adam):
    gabor_generator.apply_changes()

    optimizer = optimizer(gabor_generator.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    old_lr = lr
    lr_change_counter = 0

    pbar = trange(epochs, desc="Loss: {}".format(np.nan), leave=True, position=0)
    saved_rfs = []
    for epoch in pbar:
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if old_lr != current_lr:
            old_lr = current_lr
            lr_change_counter += 1

        if lr_change_counter > 3:
            break

        def closure():
            optimizer.zero_grad()

            # generate gabor
            gabor = gabor_generator()

            if fixed_norm is not None:
                gabor_norm = torch.norm(gabor)
                gabor_norm_constrained = fixed_norm * gabor / gabor_norm

            loss = -model_neuron(gabor_norm_constrained.cuda())

            loss.backward()

            return loss

        loss = optimizer.step(closure)

        pbar.set_description("Loss: {:.2f}".format(loss.item()))

        if save_rf_every_n_epoch is not None:
            gabor = gabor_generator()
            if (epoch % save_rf_every_n_epoch) == 0:
                saved_rfs.append(gabor.squeeze().cpu().data.numpy())

        lr_scheduler.step(-loss)

    gabor_generator.eval()
    return gabor_generator, saved_rfs
