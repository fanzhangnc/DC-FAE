from torch.nn import (
    BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, Module, PReLU, Sequential
)
from models.e4e.helpers import (
    Flatten, bottleneck_IR, bottleneck_IR_SE, get_blocks, l2_norm
)


# Modified Backbone implementation from TreB1eN (https://github.com/TreB1eN/InsightFace_Pytorch)


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        """
        Initialize the Backbone model.

        Args:
            input_size (int): Input image size (112 or 224).
            num_layers (int): Number of layers (50, 100, or 152).
            mode (str): Model mode ('ir' or 'ir_se').
            drop_ratio (float): Dropout ratio.
            affine (bool): Affine transformation in BatchNorm layers.
        """
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "Invalid input size. Choose either 112 or 224."
        assert num_layers in [50, 100, 152], "Invalid number of layers. Choose either 50, 100, or 152."
        assert mode in ['ir', 'ir_se'], "Invalid mode. Choose either 'ir' or 'ir_se'."

        blocks = get_blocks(num_layers)
        # unit_module is a reference to a class constructor (bottleneck_IR or bottleneck_IR_SE).
        unit_module = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64)
        )

        if input_size == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 7 * 7, 512),
                BatchNorm1d(512, affine=affine)
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 14 * 14, 512),
                BatchNorm1d(512, affine=affine)
            )

        # Create a list of module instances using a nested list comprehension.
        modules = [
            unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride)
            for block in blocks for bottleneck in block
        ]
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def IR_50(input_size):
    """Constructs an IR-50 model."""
    return Backbone(input_size, num_layers=50, mode='ir', drop_ratio=0.4, affine=False)


def IR_101(input_size):
    """Constructs an IR-101 model."""
    return Backbone(input_size, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)


def IR_152(input_size):
    """Constructs an IR-152 model."""
    return Backbone(input_size, num_layers=152, mode='ir', drop_ratio=0.4, affine=False)


def IR_SE_50(input_size):
    """Constructs an IR-SE-50 model."""
    return Backbone(input_size, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=False)


def IR_SE_101(input_size):
    """Constructs an IR-SE-101 model."""
    return Backbone(input_size, num_layers=100, mode='ir_se', drop_ratio=0.4, affine=False)


def IR_SE_152(input_size):
    """Constructs an IR-SE-152 model."""
    return Backbone(input_size, num_layers=152, mode='ir_se', drop_ratio=0.4, affine=False)
