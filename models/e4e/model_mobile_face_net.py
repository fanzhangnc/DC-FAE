from torch.nn import BatchNorm1d, Linear, Module
from models.e4e.helpers import ConvBlock, DepthWise, Flatten, LinearBlock, Residual, l2_norm


class MobileFaceNet(Module):
    def __init__(self, embedding_dim):
        super(MobileFaceNet, self).__init__()
        # Initial convolutional layers
        self.conv1 = ConvBlock(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        # Depth-wise and residual blocks
        self.conv_23 = DepthWise(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_blocks=4, groups=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = DepthWise(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_blocks=6, groups=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = DepthWise(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_blocks=2, groups=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Final layers before embedding
        self.conv_6_sep = ConvBlock(128, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_dim, bias=False)  # Fully connected layer
        self.bn = BatchNorm1d(embedding_dim)  # Batch normalization

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)

        # Optionally normalize the output embeddings
        # return l2_norm(x)

        return x
