import torch.nn as nn

DLparams = dict(batch_size=32)

BaselineGanHP = dict(
    dscParams=dict(spectral_norm_cond=False),
    genParams=dict(z_dim=4, featuremap_size=4, out_channels=3),
    dscLossParams=dict(label_noise=0.2, data_label=1),
    genLossParams=dict(data_label=1),
    optimizerParams=dict(type='Adam', lr=0.0002, betas=(0.5, 0.99)),
    trainBatchParams=dict(with_gradient_penalty=False)
)

SNGanHP = dict(
    dscParams=dict(spectral_norm_cond=True),
    genParams=dict(z_dim=4, featuremap_size=4, out_channels=3),
    dscLossParams=dict(label_noise=0.2, data_label=1),
    genLossParams=dict(data_label=1),
    optimizerParams=dict(type='Adam', lr=0.0002, betas=(0.5, 0.99)),
    trainBatchParams=dict(with_gradient_penalty=False)
)

WGanHP = dict(
    dscParams=dict(spectral_norm_cond=False),
    genParams=dict(z_dim=4, featuremap_size=4, out_channels=3),
    dscLossParams=dict(label_noise=0.2, data_label=1),
    genLossParams=dict(data_label=1),
    optimizerParams=dict(type='Adam', lr=0.0002, betas=(0.5, 0.99)),
    trainBatchParams=dict(with_gradient_penalty=True)
)

SNWGanHP = dict(
    dscParams=dict(spectral_norm_cond=True),
    genParams=dict(z_dim=4, featuremap_size=4, out_channels=3),
    dscLossParams=dict(label_noise=0.2, data_label=1),
    genLossParams=dict(data_label=1),
    optimizerParams=dict(type='Adam', lr=0.0002, betas=(0.5, 0.99)),
    trainBatchParams=dict(with_gradient_penalty=True)
)

