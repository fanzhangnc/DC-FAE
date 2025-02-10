from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import (
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, PReLU, ReLU,
    Sequential, Sigmoid
)


# Modified ArcFace implementation from TreB1eN
# (https://github.com/TreB1eN/InsightFace_Pytorch)

# ################################ Original ArcFace Model ################################

class Flatten(Module):
    def forward(self, input):
        # Flatten the input tensor while keeping the batch size dimension.
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    # Compute the L2 norm along the specified axis.
    norm = torch.norm(input, 2, axis, True)
    # Divide the input by the L2 norm to normalize it.
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet bottleneck block."""


def get_block(in_channel, depth, num_units, stride=2):
    """
    Create a list of Bottleneck blocks.

    Args:
        in_channel (int): Input channels.
        depth (int): Output channels.
        num_units (int): Number of Bottleneck units.
        stride (int): Stride for the first unit.

    Returns:
        list: Bottleneck blocks.
    """
    # The first block has the specified stride, the rest have stride 1
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    """
    Create a list of blocks for the network.

    Args:
        num_layers (int): Number of layers (50, 100, or 152).

    Returns:
        list: Network blocks.

    Raises:
        ValueError: If num_layers is not one of [50, 100, 152].
    """
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(f"Invalid number of layers: {num_layers}. Expected one of [50, 100, 152].")

    return blocks


class SEModule(Module):
    """
    Squeeze-and-Excitation (SE) module implementation.
    """

    def __init__(self, channels, reduction):
        """
        Initialize the SE module.

        Args:
            channels (int): Number of input and output channels.
            reduction (int): Reduction ratio for the intermediate channel size.
        """
        super(SEModule, self).__init__()

        # Global average pooling to create channel-wise statistics
        self.avg_pool = AdaptiveAvgPool2d(1)

        # First fully connected layer (implemented as 1x1 convolution)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        # ReLU activation function
        self.relu = ReLU(inplace=True)

        # Second fully connected layer (implemented as 1x1 convolution)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        # Sigmoid activation function to obtain scaling factors
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SE module.

        Args:
            x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            Tensor: Scaled input tensor.
        """
        module_input = x

        # Apply global average pooling
        x = self.avg_pool(x)

        # Pass through the first fully connected layer and ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Pass through the second fully connected layer and Sigmoid
        x = self.fc2(x)
        x = self.sigmoid(x)

        # Scale the input tensor by the learned scaling factors
        return module_input * x


class bottleneck_IR(Module):
    """
    Defines a bottleneck block for the 'IR' (Identity Residual) version of the ResNet architecture.
    """

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()

        # Define the shortcut layer
        if in_channel == depth:
            # If input and output channels are the same, use a MaxPool layer
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            # If input and output channels are different, use a Conv-BN sequence
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )

        # Define the residual layer
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth)
        )

    def forward(self, x):
        # Apply shortcut and residual layers
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut  # Combine them using element-wise addition


class bottleneck_IR_SE(Module):
    """
    Defines a bottleneck block for the 'IR_SE' (Identity Residual with Squeeze-and-Excitation) version of the ResNet architecture.
    """

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()

        # Define the shortcut layer
        if in_channel == depth:
            # If input and output channels are the same, use a MaxPool layer
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            # If input and output channels are different, use a Conv-BN sequence
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )

        # Define the residual layer with SE module
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)  # Squeeze-and-Excitation module
        )

    def forward(self, x):
        # Apply shortcut and residual layers
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut  # Combine them using element-wise addition


# ################################ MobileFaceNet ##########################################

class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        # Initialize a convolutional block with Conv2D, BatchNorm, and PReLU layers
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = BatchNorm2d(out_channels)  # Batch normalization layer
        self.prelu = PReLU(out_channels)  # PReLU activation layer

    def forward(self, x):
        # Forward pass: conv -> batch norm -> PReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class LinearBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        """
        A linear block with Conv2D and BatchNorm layers.
        """
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = BatchNorm2d(out_channels)  # Batch normalization layer

    def forward(self, x):
        """
        Forward pass through the linear block.
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWise(Module):
    def __init__(self, in_channels, out_channels, residual=False, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 groups=1):
        """
        Depthwise separable convolution block with optional residual connection.
        """
        super(DepthWise, self).__init__()
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=groups,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )
        self.conv_dw = ConvBlock(
            in_channels=groups,
            out_channels=groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        self.project = LinearBlock(
            in_channels=groups,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x  # Save input for residual connection
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x  # Add the residual
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, channels, num_blocks, groups, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        """
        Residual block that stacks multiple depth-wise separable convolutions.
        """
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(
                DepthWise(
                    in_channels=channels,
                    out_channels=channels,
                    residual=True,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
