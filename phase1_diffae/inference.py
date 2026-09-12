import torch


def generate_face(model, conf, device, z, x_T=None, T=None):
    """
    Generate a face from a DiffAE 512-dimensional latent/style code.

    Parameters
    ----------
    model:
        Loaded pretrained DiffAE model.

    conf:
        DiffAE TrainConfig.

    device:
        torch.device.

    z:
        Tensor with shape (batch_size, 512).

    x_T:
        Optional diffusion noise tensor.
        If omitted, a stochastic x_T is generated.

    T:
        Diffusion sampling steps.
        Defaults to the configuration's evaluation T.
    """

    if T is None:
        T = conf.T_eval

    if z.ndim != 2 or z.shape[1] != conf.style_ch:
        raise ValueError(
            f"Expected z shape (batch, {conf.style_ch}), "
            f"but received {tuple(z.shape)}"
        )

    z = z.to(device)

    if x_T is None:
        x_T = torch.randn(
            z.shape[0],
            3,
            conf.img_size,
            conf.img_size,
            device=device,
        )
    else:
        x_T = x_T.to(device)

    with torch.no_grad():
        generated = model.render(
            noise=x_T,
            cond=z,
            T=T,
        )

    return generated.clamp(0, 1)