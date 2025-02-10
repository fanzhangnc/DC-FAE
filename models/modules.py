# Standard library imports
import copy
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision

# Local application imports
from .flows.realnvp import RealNVP


def mlp(in_c, hid_c, out_c, n_layer=0):
    """
    Defines a multi-layer perceptron (MLP) with specified layers.

    Args:
        in_c (int): Input dimension.
        hid_c (int): Hidden layer dimension.
        out_c (int): Output dimension.
        n_layer (int): Number of hidden layers (default is 0).

    Returns:
        nn.Sequential: A sequential container with the defined layers.
    """
    layers = [
        nn.Linear(in_c, hid_c),  # Input layer
        nn.BatchNorm1d(hid_c),  # Batch normalization
        nn.ReLU(True)  # ReLU activation
    ]

    # Add hidden layers
    for _ in range(n_layer):
        layers += [
            nn.Linear(hid_c, hid_c),
            nn.BatchNorm1d(hid_c),
            nn.ReLU(True)
        ]

    # Output layer
    layers.append(nn.Linear(hid_c, out_c))

    return nn.Sequential(*layers)


class Classifier(nn.Module):
    """
    Classifier using ResNet backbone for feature extraction and multi-head classification tasks.
    """

    def __init__(self, backbone='r34'):
        """
        Initialize classifier with a specified ResNet backbone.

        Args:
            backbone (str): ResNet backbone type ('r34' or 'r50').
        """
        super().__init__()

        # Load the appropriate ResNet backbone
        if backbone == 'r34':
            backbone = torchvision.models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
            in_dim = 512
        elif backbone == 'r50':
            backbone = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            in_dim = 2048
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

        # Feature extractor without the final fully connected layer
        self.extractor = copy.deepcopy(backbone)
        self.extractor.fc = nn.Identity()

        # Attribute heads (40 attributes)
        self.attr_heads = nn.ModuleList([mlp(in_dim, in_dim, 1, 2) for _ in range(40)])

        # Age head (6 output classes)
        self.age_heads = copy.deepcopy(backbone)
        self.age_heads.fc = nn.Linear(in_dim, 6)

    def forward(self, x):
        """
        Forward pass for both attributes and age predictions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attribute and age predictions.
        """
        features = self.extractor(x)

        # Get attribute predictions
        outs = [head(features) for head in self.attr_heads]

        # Get age predictions
        outs.append(self.age_heads(x))

        # Concatenate all outputs
        outs = torch.cat(outs, dim=1)

        # Apply sigmoid and threshold to get binary predictions
        preds = (torch.sigmoid(outs) > 0.5).float()

        # Aggregate age group predictions
        preds = torch.cat([preds[:, :40], preds[:, 40:].sum(dim=1).unsqueeze(1)], dim=1)

        return outs, preds

    def forward_attr(self, x):
        """
        Forward pass for attribute predictions only.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attribute outputs and predictions.
        """
        features = self.extractor(x)
        attr_outs = [head(features) for head in self.attr_heads]
        outs = torch.cat(attr_outs, dim=1)
        preds = (torch.sigmoid(outs) > 0.5).float()
        return outs, preds

    def forward_age(self, x):
        """
        Forward pass for age prediction only.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Age output and prediction.
        """
        outs = self.age_heads(x)
        preds = (torch.sigmoid(outs) > 0.5).float().sum(dim=1)
        return outs, preds


class FLOW(nn.Module):
    def __init__(self, style_dim=2, n_layer=10, n_styles=18):
        super(FLOW, self).__init__()
        self.flow = RealNVP(style_dim, n_layer)

    def forward(self, inputs):
        if inputs.ndim == 3:
            bs, n_styles, dim = inputs.size()
            latents = inputs.view(bs * n_styles, -1)
        elif inputs.ndim == 2:
            bs, dim = inputs.size()
            latents = inputs
        else:
            raise NotImplementedError
        z, log_det_jacobian = self.flow(latents)
        logz_sum = D.Normal(0, 1).log_prob(z).sum(dim=1)
        loss = - torch.mean(logz_sum + log_det_jacobian) / dim
        logz = logz_sum / dim
        return loss, logz, log_det_jacobian, z

    def backward(self, inputs):
        bs, n_styles, dim = inputs.size()
        inputs = inputs.view(bs * n_styles, -1)
        return self.flow.backward(inputs)[0].view(-1, n_styles, dim)


class Affine(nn.Module):
    """
    Defines an affine transformation with learnable scale and bias based on condition input.

    Args:
        dim_out (int): Output dimension.
        dim_c (int): Condition input dimension.
    """

    def __init__(self, dim_out, dim_c):
        super(Affine, self).__init__()

        # MLPs for calculating scale and bias
        self.scale = mlp(dim_c, dim_out, dim_out)
        self.bias = mlp(dim_c, dim_out, dim_out)

    def forward(self, x, c):
        """
        Apply affine transformation to the input tensor.

        Args:
            x (Tensor): Input tensor.
            c (Tensor): Condition tensor.

        Returns:
            Tensor: Transformed output after applying affine transformation.
        """
        # Compute scale and apply exp to ensure positivity
        scale = self.scale(c).exp()

        # Compute bias
        bias = self.bias(c)

        # Apply affine transformation
        return x * scale + bias


class ConcatSquashLinear(nn.Module):
    """
    Defines the ConcatSquashLinear class which contains two submodules: `mlp` and `Affine`.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        dim_c (int): Condition input dimension.
    """

    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()

        # Define the submodules
        self.layer = mlp(dim_in, dim_in, dim_out, 0)
        self.affine = Affine(dim_out, dim_c)

    def forward(self, x, c):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.
            c (Tensor): Condition tensor.

        Returns:
            Tensor: Output after applying MLP and affine transformation.
        """
        x = self.layer(x)
        x = self.affine(x, c)
        return x


# TransformerModel class definition
class TransformerModel(nn.Module):
    def __init__(self, hid_c, d_model, nhead, num_layers):
        """
        Initialize TransformerModel module.

        Args:
            hid_c (int): Hidden layer feature dimension.
            d_model (int): Input and output feature dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
        """
        super(TransformerModel, self).__init__()

        # Store hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Initialize position encoder and transformer encoder
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Linear layer
        self.encoder = nn.Linear(d_model, hid_c)

    def forward(self, trajectory):
        """
        Forward pass.

        Args:
            trajectory (torch.Tensor): Input tensor with shape (sequence_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor after transformer encoding with shape (sequence_len, batch_size, hid_c).
        """
        # Add position encoding and pass through transformer
        trajectory_emb = self.pos_encoder(trajectory)
        y = self.transformer_encoder(trajectory_emb)

        # Apply linear layer
        y = self.encoder(y)
        return y


# Positional Encoding class definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model (int): The feature dimension of input and output.
            max_len (int): Maximum length of the position encoding (default is 5000).
        """
        super(PositionalEncoding, self).__init__()

        # Create a tensor of zeros to store position encoding with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the div_term for sinusoidal function
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the tensor and register it as a buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StyleLSTM(nn.Module):
    def __init__(self, layer, style_dim, hid_c, c_dim, n_styles, n_layer=0):
        super(StyleLSTM, self).__init__()

        # Initialize the main layers
        layers = [layer(hid_c, hid_c, c_dim)]
        for _ in range(n_layer - 1):
            layers.append(layer(hid_c, hid_c, c_dim))
        self.main = nn.ModuleList(layers)

        # Define the input, output, and step layers
        self.input_layer = mlp(style_dim * n_styles, hid_c, hid_c)
        self.output_layer = mlp(hid_c, hid_c, style_dim)
        self.step_layer = mlp(hid_c, hid_c, 1)

        # Define LSTM layers
        self.rnn_layers = 2
        self.rnn = nn.LSTM(hid_c, hid_c, self.rnn_layers)

        self.n_styles = n_styles
        self.style_dim = style_dim

    def forward(self, trajectory, c):
        # Flatten trajectory and process through the input layer
        trajectory = torch.stack(trajectory).flatten(-2)
        y = self.input_layer(trajectory.view(-1, trajectory.size(-1)))
        y = y.view(trajectory.size(0), trajectory.size(1), -1)

        # Process through LSTM layers
        h_0 = torch.zeros(self.rnn_layers, y.size(1), y.size(2)).to(y)
        c_0 = torch.zeros(self.rnn_layers, y.size(1), y.size(2)).to(y)
        y, _ = self.rnn(y, (h_0, c_0))

        # Pass through the main layers
        y0 = y[-1]
        for layer in self.main:
            y0 = layer(y0, c)

        # Compute step size and normalize output
        step_size = torch.sigmoid(self.step_layer(y0))
        dy = F.normalize(self.output_layer(y0), dim=-1).unsqueeze(1)

        return dy, step_size


class StyleTransformer(nn.Module):
    # layer: Base layer class, style_dim: Style vector dimension, hid_c: Hidden layer size,
    # c_dim: Conditioning dimension, n_styles: Number of styles,
    # n_layer: Number of main layers (default 0)
    def __init__(self, layer, style_dim, hid_c, c_dim, n_styles, transformer_params, n_layer=0):
        super(StyleTransformer, self).__init__()

        # Initialize the main layers
        layers = [layer(hid_c, hid_c, c_dim)]
        for _ in range(n_layer - 1):
            layers.append(layer(hid_c, hid_c, c_dim))
        self.main = nn.ModuleList(layers)

        # Initialize Transformer parameters
        self.d_model = transformer_params['d_model']
        self.nhead = transformer_params['nhead']
        self.num_layers = transformer_params['num_layers']

        # Define input, output, and step layers
        self.input_layer = mlp(style_dim * n_styles, hid_c, self.d_model)
        self.output_layer = mlp(hid_c, hid_c, style_dim)
        self.step_layer = mlp(hid_c, hid_c, 1)

        # Initialize Transformer model
        self.transformer = TransformerModel(hid_c, self.d_model, self.nhead, self.num_layers)

        self.n_styles = n_styles
        self.style_dim = style_dim

    def forward(self, trajectory, c):
        # trajectory: List of tensors with shape [256, 18, 512]

        # Stack the list into a single tensor and flatten the last two dimensions
        trajectory = torch.stack(trajectory, dim=0).flatten(-2)  # Shape: [5, 256, 9216]

        # Reshape the trajectory for input into the input layer
        flattened_trajectory = trajectory.view(-1, trajectory.size(-1))

        # Pass the reshaped trajectory through the input layer
        y = self.input_layer(flattened_trajectory)  # Shape: [1280, 256]

        # Reshape y to match the transformer's input dimensions
        y = y.view(trajectory.size(0), trajectory.size(1), -1)  # Shape: [5, 256, 256]

        # Pass through the transformer
        y = self.transformer(y)  # Shape: [5, 256, 512]

        # Take the last element in the sequence (final output of the transformer)
        y0 = y[-1]

        # Pass through the main layers
        for layer in self.main:
            y0 = layer(y0, c)

        # Compute the step size
        step_size = torch.sigmoid(self.step_layer(y0))

        # Normalize the output and expand dimensions
        dy = F.normalize(self.output_layer(y0), dim=-1).unsqueeze(1)

        return dy, step_size


# Class definition
class ANT(nn.Module):
    def __init__(self,
                 c_dim=4,
                 hid_dim=512,
                 max_steps=20,
                 n_layers=10,
                 n_styles=18,
                 style_dim=512,
                 transformer_params=None):
        super(ANT, self).__init__()

        self.n_layers = n_layers
        self.style_dim = style_dim
        self.c_dim = c_dim
        self.max_steps = max_steps

        # Initialize the style model (either Transformer or LSTM)
        if transformer_params:
            self.style_model = StyleTransformer(
                ConcatSquashLinear, style_dim, hid_dim, c_dim,
                n_styles, transformer_params, self.n_layers
            )
        else:
            self.style_model = StyleLSTM(
                ConcatSquashLinear, style_dim, hid_dim, c_dim,
                n_styles, self.n_layers
            )

    def forward(self, x, target):
        target = target.float()

        y0 = x
        step_sizes = []
        trajectory = [y0]

        # Iterate through the maximum number of steps
        for t in range(self.max_steps):
            dy, step_size = self.style_model(trajectory, target)

            # Update the current state
            y1 = y0 + dy * step_size[:, :, None]
            y0 = y1

            # Store step size and update trajectory
            step_sizes.append(step_size)
            trajectory.append(y0)

        # Concatenate step sizes and stack trajectory
        step_sizes = torch.cat(step_sizes, dim=1)
        trajectory = torch.stack(trajectory)

        return trajectory, step_sizes


class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    During the forward pass, it acts as an identity function.
    During the backward pass, it multiplies the gradient by -1.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator(nn.Module):
    """
    Discriminator model for domain adaptation.
    Consists of three convolutional layers.
    """

    def __init__(self, k=2, pool_out="avg", pool_output_size=512):
        super(Discriminator, self).__init__()
        self.pooling_layer = PoolingLayer(pool_out, pool_output_size)
        output_size = pool_output_size

        self.conv1 = nn.Conv2d(
            in_channels=output_size, out_channels=output_size // k,
            kernel_size=1, stride=1, bias=True
        )
        output_size = output_size // k

        self.conv2 = nn.Conv2d(
            in_channels=output_size, out_channels=output_size // k,
            kernel_size=1, stride=1, bias=True
        )
        output_size = output_size // k

        self.conv3 = nn.Conv2d(
            in_channels=output_size, out_channels=2,
            kernel_size=1, stride=1, bias=True
        )

    def forward(self, x):
        x = self.pooling_layer(x)  # Apply pooling before adding dimensions
        x = x[:, :, None, None]  # Add dimensions for height and width
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)  # No activation after the last layer
        out = out.view(out.size(0), -1)  # Flatten the output
        return out


class Discriminators(nn.Module):
    """
    Container for multiple Discriminator models.
    Applies gradient reversal if specified.
    """

    def __init__(self, output_dims, grl):
        """
        Initialize Discriminators.
        """
        super(Discriminators, self).__init__()
        self.discriminators = nn.ModuleList([Discriminator() for dim in output_dims])
        self.grl = grl

    def forward(self, x):
        """
        Forward pass through the Discriminators.
        """
        if self.grl:
            # Apply gradient reversal
            out = [discriminator(grad_reverse(x[i])) for i, discriminator in enumerate(self.discriminators)]
        else:
            out = [discriminator(x[i]) for i, discriminator in enumerate(self.discriminators)]

        # Stack the outputs and return
        return torch.stack(out, dim=0)


class PoolingLayer(nn.Module):
    def __init__(self, pool_out: str, output_size: int):
        """
        Initialize with the specified pooling type and output size.

        Args:
            pool_out: "avg" for average pooling or "max" for max pooling.
            output_size: Target output size of the pooling layer.
        """
        super(PoolingLayer, self).__init__()
        pool_out = pool_out.lower()
        if pool_out == "avg":
            self.layer = nn.AdaptiveAvgPool1d(output_size)
        elif pool_out == "max":
            self.layer = nn.AdaptiveMaxPool1d(output_size)
        else:
            raise ValueError(f"Invalid pooling type: {pool_out}. Expected 'avg' or 'max'.")

        self.pool_out = pool_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling and return the processed tensor.

        Args:
            x: Input tensor of shape [batch_size, channels, seq_length].

        Returns:
            Pooled and squeezed output tensor.
        """
        x = self.layer(x)
        if self.pool_out == "max":
            x, _ = x.max(dim=1)
        else:
            x = x.mean(dim=1)
        return x.squeeze()
