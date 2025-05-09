import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime

from models.e4e.model_irse import Backbone
from models.e4e.model_mobile_face_net import MobileFaceNet
from .insightface import iresnet50, iresnet34
from . import resize

# Path to the model file for MobileFaceNet
MODEL_PATH_MOBILEFACENET = '/root/autodl-tmp/DC-FAE/data/model_mobilefacenet.pth'


def initialize_facenet(backbone):
    """
    Initialize the FaceNet model with the specified backbone.

    Args:
        backbone (str): The backbone model to initialize.

    Returns:
        The initialized model.
    """
    if backbone == 'mobilefacenet':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load(MODEL_PATH_MOBILEFACENET))
        return model

    if backbone == 'r50':
        return iresnet50(pretrained=True)

    return iresnet34(pretrained=True)


def resize_tensor(x):
    """
    Resize the input tensor to 112x112 if necessary.

    Args:
        x (torch.Tensor): Input images tensor.

    Returns:
        torch.Tensor: Resized images tensor.
    """
    if x.size(-1) == 112:
        return x
    return resize(x, 112)


def crop_and_resize(x):
    """
    Crop and resize the input images.

    Args:
        x (torch.Tensor): Input images tensor.

    Returns:
        torch.Tensor: Cropped and resized images tensor.
    """
    w, h = x.size(-2), x.size(-1)
    assert w == h, "Width and height must be equal for cropping"
    img_size = w
    scale = lambda x: int(x * img_size / 256)
    h, x1, x2 = scale(188), scale(35), scale(32)
    x = x[:, :, x1:x1 + h, x2:x2 + h]
    return resize_tensor(x)


class IDLoss(nn.Module):
    def __init__(self, crop=False, backbone='mobilefacenet'):
        super(IDLoss, self).__init__()
        self.facenet = initialize_facenet(backbone)
        self.crop = crop
        self.embeddings = None
        self.facenet.eval()

    @torch.no_grad()
    def extract_dataset(self, loader):
        """
        Extract features from the entire dataset.

        Args:
            loader (DataLoader): DataLoader for the dataset.

        Returns:
            None
        """
        embeddings = []
        for inputs in loader:
            images = inputs[0].cuda()
            embeddings.append(F.normalize(self.extract_features(images), dim=1))
        self.embeddings = torch.cat(embeddings, dim=0)

    def extract_features(self, x):
        """
        Extract features from the input images.

        Args:
            x (torch.Tensor): Input images tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        if self.crop:
            x = crop_and_resize(x)
        else:
            x = resize_tensor(x)
        return self.facenet(x)

    def forward(self, input, recon):
        """
        Calculate the identity loss.

        Args:
            input_tensor: Input images tensor.
            recon: Reconstructed images tensor.

        Returns:
            Identity loss.
        """
        input = input * 0.5 + 0.5  # Normalize to [0, 1]
        recon = recon * 0.5 + 0.5  # Normalize to [0, 1]

        with torch.no_grad():
            e1 = F.normalize(self.extract_features(input), dim=1)
        e2 = F.normalize(self.extract_features(recon), dim=1)

        identity_loss = - (e1 * e2).sum(dim=1).mean()
        return identity_loss


class BetweenLoss(nn.Module):
    """
    Custom loss function that combines multiple losses with given weights.
    """

    def __init__(self, gamma=None, loss=nn.MSELoss(reduction='none')):
        super(BetweenLoss, self).__init__()
        self.gamma = gamma if gamma is not None else [1] * 10  # Adjust gamma length to match the steps
        self.loss = loss

    def forward(self, outputs, targets, weights):
        """
        Forward pass for the custom loss function.
        """
        assert outputs.shape == targets.shape, "Outputs and targets must have the same shape."

        return self.calculate_total_loss(outputs, targets, weights)

    def calculate_total_loss(self, outputs, targets, weights):
        """
        Calculate the total loss.
        """
        total_loss = 0
        steps = outputs.size(0)

        for i in range(steps):
            mse = self.loss(outputs[i], targets[i])  # Shape: [batch_size, num_layers, latent_dim]

            # Expand weights to match the shape of mse
            # w_exp = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
            w_exp = weights.unsqueeze(-1)  # Shape: [batch_size, 1, 1]

            # print('w_exp.shape', w_exp.shape)

            # Multiply each element of the loss by the corresponding weight
            weighted_mse = mse * w_exp  # Shape: [batch_size, num_layers, latent_dim]

            # Compute the mean over all dimensions to get the scalar loss
            avg_weighted_mse = weighted_mse.mean()  # Scalar
            total_loss += self.gamma[i] * avg_weighted_mse

        return total_loss


class BetweenFakeLoss(nn.Module):
    """
    Placeholder loss function for cases where no actual between loss is needed.
    """

    def forward(self, outputs, targets, weights):
        # Return a scalar zero loss
        return (0 * outputs[0]).sum()


class DiscriminatorLoss(nn.Module):
    """
    Loss function for discriminators in domain adaptation.
    Combines multiple discriminator outputs with given weights.
    """

    def __init__(self, models, eta=None):
        super(DiscriminatorLoss, self).__init__()
        self.models = models
        self.eta = eta if eta is not None else [1] * 10  # Adjust gamma length to match the steps
        # self.loss = loss

    def forward(self, outputs, targets, loss):
        """
        Forward pass for the custom discriminator loss function.
        """
        steps = outputs.size(0)
        batch_size = outputs.size(1)

        # Concatenate each pair of output and target tensors along the batch dimension
        inputs = torch.cat((outputs, targets), dim=1)

        # Create the target tensor for binary classification
        target = torch.FloatTensor(
            [[1, 0] for _ in range(batch_size)] + [[0, 1] for _ in range(batch_size)]
        ).to(inputs.device)

        # Pass the concatenated inputs through the models
        outputs = self.models(inputs)

        # Compute the weighted sum of losses for each model output
        res = sum(self.eta[i] * loss(outputs[i], target) for i in range(steps))

        return res


class DiscriminatorFakeLoss(nn.Module):
    """
    Placeholder loss function for cases where no actual discriminator loss is needed.
    """

    def forward(self, outputs, targets, loss):
        return (0 * outputs[0]).sum()
