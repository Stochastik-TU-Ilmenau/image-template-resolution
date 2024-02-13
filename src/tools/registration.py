"""
Helper functions to create an affine or rigid template for a set of images, also returns the registred images (2d and 3d)
"""

import matplotlib.pyplot as plt
import torch as t
t.set_default_dtype(t.float32)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def affine_transform(moving, affine, mode=None):
    # moving: image to be transformed (only 2d and 3d)
    # affine: affine map for transformation

    shape = moving.shape

    if mode == 'rigid': # quick way of getting a rotation matrix parametrized by "affine" parameter
        A = affine[:, :-1]
        R = t.matrix_exp(A - A.T)
        affine = t.column_stack([R, affine[:, -1]])

    # pytorch uses the center of the image as origin for the affine map!
    # pytorch expects tensors with shape (N, C, H, W) or (N, C, D, H, W)
    # need to add N, C dimensions!
    affine_grid = t.nn.functional.affine_grid(affine[None,], (1, 1) + shape, align_corners=False)
    grid_samples = t.nn.functional.grid_sample(moving[None, None, ...], affine_grid, padding_mode='zeros', align_corners=False)
    return grid_samples[0, 0, ...] # remove N, C dimensions


def aff_inv(aff):
    # (A^(-1), - A^(-1) b) is the inverse of the affine map aff=(A, b)
    b = aff[:, -1]
    Ainv = t.inverse(aff[:, :-1])
    return t.column_stack([Ainv, Ainv @ -b])


def data_loss(template_warped, target, norm='l2'):
    if norm == 'l2':
        return t.mean((template_warped - target)**2)
    if norm == 'l1':
        return t.nn.functional.l1_loss(template_warped, target)
    else:
        ValueError('Only l2 or l1 norm is implemented!')


def registration(images, mode='affine', max_iter=200, scale_intensities=True, data_norm='l2', plot_template=lambda *_:None, one_dim=False, lr=0.01):
    n = len(images)
    nd = images.dim() - 1

    # affine maps initial
    aff_id = t.eye(nd + 1)[:nd,].to(device)
    affine_maps = aff_id.repeat(n, 1, 1)
    affine_maps.requires_grad = True

    # initial guess for template
    template = images.mean(axis=0).to(device) 
    template.requires_grad = True

    # intensity rescaling factor per image
    intensity_rescale = t.ones(n)
    intensity_rescale.requires_grad = True

    # optimize template and maps simultaneously
    optimizer = t.optim.Adam([template, affine_maps, intensity_rescale], lr=lr)


    losses = []
    aff_norm = []
    for iter in range(1, max_iter + 1):

        loss_cum = 0.0
        optimizer.zero_grad()

        for target, aff, s in zip(images, affine_maps, intensity_rescale):

            if one_dim: # only keep 1d part of 2d affine transformation
                #
                #                      (* * *)   (0 0 0)   (1 0 0)   (1 0 0)
                # aff' = aff * M + N = (* * *) * (0 1 1) + (0 0 0) = (0 * *)
                #
                M, N = aff_id.clone(), aff_id.clone()
                M[0, 0] = 0.0
                M[1, 2] = 1.0
                N[1, 1] = 0.0
                aff = aff * M + N

            template_warped = affine_transform(template, aff, mode)
            if scale_intensities:
                template_warped = s * template_warped

            loss = data_loss(template_warped, target.to(device), norm=data_norm) + 1.0 * t.mean((aff - aff_id)**2) + 0.01 * (s - 1)**2
            
            loss.backward()  # gradient accumulation!
            loss_cum += loss.detach().cpu().numpy()

        optimizer.step()

        losses.append(loss_cum)
        aff_norm.append(affine_maps.norm().detach().cpu().numpy())
        template_norm = template.grad.data.norm().cpu().numpy()

        if iter % 10 == 0:
            print('RIGID REGISTRATION' if mode == 'rigid' else 'AFFINE REGISTRATION')
            print('norm for datafit:', data_norm)
            print('iter:', iter, 'max_iter:', max_iter)
            print('loss:', losses[-1], 'template grad norm:', template_norm)

            plt.plot(losses)
            plt.show()

            plt.plot(aff_norm)
            plt.show()

            plot_template(template)

            #print(intensity_rescale)
    

    print('register individuals with template ...')
    # register individual images with the template using the inverse affine maps:
    affine_maps.requires_grad = False # saves memory
    images_reg = t.empty_like(images)
    for i, (brain, aff, s) in enumerate(zip(images, affine_maps, intensity_rescale)):
        warped = affine_transform(brain.to(device), aff_inv(aff))
        images_reg[i,] = 1 / s * warped

    return template.detach().cpu().numpy(), affine_maps.detach().cpu().numpy(), images_reg.detach().cpu().numpy(), intensity_rescale.detach().cpu().numpy()

