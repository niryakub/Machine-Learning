import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device) # used in line 6. in the training algorithm
    interpolated_images = real * epsilon + fake * (1 - epsilon) # ^^ (according to prop.1 in paper)

    # Calculating critic scores:
    mixed_scores = critic(interpolated_images) # line 7. Dw(x^)

    # Calculating Gradient of Dw(x^) wrt to x^:
    gradient = torch.autograd.grad( # .grad = Computes and returns the sum of gradients of outputs wrt the inputs (in our case, interpolated_images).
        # unlike .backward which computes w.r.t graph leaves.
        # i.e. if we did out.backward() for some variable out that involved x in it's calculations, then x.grad will hold d-out/d-x.
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1) # flatten (grad.shape[0] = num of samples)
    gradient_norm = gradient.norm(2, dim=1) # taking L2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2) # Line 7.
    return gradient_penalty