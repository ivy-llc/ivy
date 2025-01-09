from .helpers import (
    resnet18,
    resnet34,
    resnet50,
    resnext50_32x4d,
    wide_resnet50_2,
    freia_cflow_head,
    freia_flow_head,
    positionalencoding2d,
)
import torch
import numpy as np


def load_encoder_arch(c, L):
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ["layer" + str(i) for i in range(L)]
    if "resnet" in c.enc_arch:
        if c.enc_arch == "resnet18":
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == "resnet34":
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == "resnet50":
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == "resnext50_32x4d":
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == "wide_resnet50_2":
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError(
                "{} is not supported architecture!".format(c.enc_arch)
            )
        #
        if L >= 3:
            """hooks not supported yet in the frontends.."""
            # encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if "wide" in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            """hooks not supported yet in the frontends.."""
            # encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if "wide" in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            """hooks not supported yet in the frontends.."""
            # encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if "wide" in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1

    return encoder, pool_layers, pool_dims


def load_decoder_arch(c, dim_in):
    if c.dec_arch == "freia-flow":
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == "freia-cflow":
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError("{} is not supported NF!".format(c.dec_arch))
    return decoder


def encoder_inference(encoder, image):
    return encoder(image)


def decoder_inference(decoders, activations, c):
    P = c.condition_vec
    N = 256
    for l, activation in enumerate(activations):
        if "vit" in c.enc_arch:
            e = activation.transpose(1, 2)[..., 1:]
            e_hw = int(np.sqrt(e.size(2)))
            e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
        else:
            e = activation  # BxCxHxW
        #
        B, C, H, W = e.size()
        S = H * W
        E = B * S
        #
        p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
        e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
        perm = torch.randperm(E).to(c.device)  # BHW
        decoder = decoders[l]
        #
        FIB = E // N  # number of fiber batches
        assert (
            FIB > 0
        ), "MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!"
        for f in range(FIB):  # per-fiber processing
            idx = torch.arange(f * N, (f + 1) * N)
            c_p = c_r[perm[idx]]  # NxP
            e_p = e_r[perm[idx]]  # NxC
            if "cflow" in c.dec_arch:
                z, log_jac_det = decoder(
                    e_p,
                    [
                        c_p,
                    ],
                )
            else:
                z, log_jac_det = decoder(e_p)

            return z, log_jac_det
