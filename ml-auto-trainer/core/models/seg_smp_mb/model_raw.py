import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



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



class SmpMultibranch(nn.Module):

    MODE_FUSE_ALL = 0
    MODE_FUSE_LAST = 1

    def __init__(self, encoder_name, decoder_name, in_channels_list: list, out_channels, activation, fuse_mode =MODE_FUSE_ALL):
        self.fuse_mode = fuse_mode

        self.branch_count = len(in_channels_list)

        # ---- Depth of the encoder block
        self.encoder_depth = None

        # ---- List of encoder blocks
        self.encoder_block_list = []
        for i in range(0,self.branch_count):
            in_channel = in_channels_list[i]
            encoder = smp.encoders.get_encoder(name=encoder_name, in_channels=in_channel, depth=5, weights=None)
            self.encoder_list.append(encoder)

            if self.feature_block_count is not None:
                self.feature_block_count = encoder.out_channels

        decoder_channels = (256, 128, 64, 32, 16)
        decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=channels,
            decoder_channels=decoder_channels
        )

        segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, tensor_list):
        encoder_output_list = []

