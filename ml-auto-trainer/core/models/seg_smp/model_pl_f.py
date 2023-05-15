import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl



encoder_a = smp.encoders.get_encoder(name="resnet34", in_channels=3, depth=5, weights=None)
encoder_b = smp.encoders.get_encoder(name="resnet34", in_channels=3, depth=5, weights=None)

x = torch.zeros(8, 3, 512, 512)
y_a = encoder_a(x)
y_b = encoder_b(x)
print([t.size() for t in y_b])

Y = []
channels = []
for i in range(0, len(y_a)):
    xx = torch.concat([y_a[i], y_b[i]], 1)
    Y.append(xx)
    channels.append(xx.size()[1])


print(encoder_a.out_channels)
decoder_channels = (256, 128, 64, 32, 16)

print(channels)
decoder = smp.decoders.unet.decoder.UnetDecoder(
    encoder_channels=channels,
    decoder_channels=decoder_channels
)

z = decoder(*Y)
print(z.size())
activation = None

segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )

z = segmentation_head(z)
print(z.size())

"""
self.encoder = get_encoder(
    encoder_name,
    in_channels=in_channels,
    depth=encoder_depth,
    weights=encoder_weights,
)

self.decoder = UnetDecoder(
    encoder_channels=self.encoder.out_channels,
    decoder_channels=decoder_channels,
    n_blocks=encoder_depth,
    use_batchnorm=decoder_use_batchnorm,
    center=True if encoder_name.startswith("vgg") else False,
    attention_type=decoder_attention_type,
)

self.segmentation_head = SegmentationHead(
    in_channels=decoder_channels[-1],
    out_channels=classes,
    activation=activation,
    kernel_size=3,
)
"""
