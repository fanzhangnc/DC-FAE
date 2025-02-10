# Standard library imports
import argparse
import os
import sys

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Other third-party imports
import tqdm

sys.path.insert(0, '/root/autodl-tmp/SL-KD')
sys.path.insert(0, '/root/autodl-tmp/SL-KD/models/stylegan2')

from models.decoder import StyleGANDecoder
from models.modules import ANT, Classifier
from models.ops import load_network

torch.autograd.set_grad_enabled(False)

# Set print options for torch tensors
torch.set_printoptions(precision=3, sci_mode=False)

# ================= Arguments ================ #

parser = argparse.ArgumentParser(description='Configuration')

# General configuration
parser.add_argument('--run_name', default=None, type=str, help='Name of the model to load')
parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID to use')
parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')

# Training configuration
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps for training process')

# Transformer configuration
parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')

# Miscellaneous
parser.add_argument('--changes', nargs='+', type=int, default=[15], help='List of change values')
parser.add_argument('--keeps', nargs='+', type=int, default=[20, -1], help='List of keep values')

opts = parser.parse_args()

# Create the dictionary
transformer_params = {
    'd_model': opts.d_model,
    'nhead': opts.nhead,
    'num_layers': opts.num_layers
}

# Print the values of hyperparameters provided by the user in the command line or the default values
print(opts)

# ================= Initialization ================ #

# Set device to GPU or CPU
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

# ================= Initialization ================ #

print('Loading checkpoints and data paths...')

# Paths to checkpointsR
stylegan2_checkpoint = '/root/autodl-tmp/SL-KD/data/ffhq.pkl'
classifier_checkpoint = '/root/autodl-tmp/SL-KD/data/focal_loss_r34_age_8410.pth'
latents_checkpoint = '/root/autodl-tmp/SL-KD/data/ffhq_train_latents.pth'
preds_checkpoint = '/root/autodl-tmp/SL-KD/data/focal_loss_ffhq_train_preds.pth'

print('Initializing StyleGAN Decoder...')
# Initialize StyleGAN Decoder
output_size = 256
G = StyleGANDecoder(
    stylegan2_checkpoint,
    start_from_latent_avg=False,
    output_size=output_size
)

print('Initializing image classifier...')
# Initialize image classifier
image_classifier = Classifier().cuda()
image_classifier.load_state_dict(load_network(torch.load(classifier_checkpoint, map_location='cpu')))
image_classifier = image_classifier.eval().cuda()

print('Loading latents and predictions...')
# Load latents and predictions
all_latents = torch.load(latents_checkpoint, map_location='cpu') + G.latent_avg
all_preds = torch.load(preds_checkpoint, map_location='cpu')

print('All data loaded successfully.')

# Move model to GPU and set it to evaluation mode
G = G.cuda().eval()


def to_distribution(prob):
    """
    Convert probabilities to a distribution suitable for KL divergence calculation.

    Args:
        prob (Tensor): Input probabilities.

    Returns:
        Tensor: Transformed distribution.
    """
    distribution = prob.unsqueeze(-1)
    distribution = torch.cat((distribution, (1 - distribution)), dim=-1)
    distribution = distribution.float()
    return distribution


def kl_divergence(input, target):
    """
    Calculate KL divergence between input and target distributions.

    Args:
        input (Tensor): Input distribution.
        target (Tensor): Target distribution.

    Returns:
        Tensor: KL divergence.
    """
    input = to_distribution(input)
    target = to_distribution(target)
    kl = F.kl_div(input.log(), target, reduction='none')
    kl = torch.sum(kl, dim=-1)
    if kl.ndim > 1:
        kl = torch.mean(kl, dim=1)
    return kl


class DualAdaTransEditor(object):
    """
    Class for managing and manipulating the latent space of dual models.
    """

    def __init__(self, checkpoints, attr_num, scale=1.0):
        """
        Initialize the DualAdaTransEditor.

        Args:
            checkpoints (list of str): Paths to the checkpoint files.
            attr_num (int): Attribute ID, used for attribute editing.
            scale (float): Scaling factor for model steps.
        """
        self.attr_num = attr_num
        self.scale = scale

        print(f'Loading models from checkpoints: {checkpoints}')
        # Initialize and load models
        self.model_s = self._load_model(checkpoints[0])
        self.model_t = self._load_model(checkpoints[1])
        print('Models loaded successfully.')

    def _load_model(self, checkpoint_path):
        """
        Load the model from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            nn.Module: The loaded and initialized model.
        """
        model = ANT(
            c_dim=c_dim,
            hid_dim=512,
            max_steps=opts.max_steps,
            n_layers=10,
            n_styles=G.n_styles,
            style_dim=G.style_dim,
            transformer_params=transformer_params
        ).eval().cuda()

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(load_network(state_dict))
        model.max_steps = int(self.scale)

        return model

    def _process_model(self, model, inputs, targets):
        """
        Process the model to get new styles and attribute probabilities.

        Args:
            model (nn.Module): The model.
            inputs (Tensor): The input latent space.
            targets (Tensor): The target attributes.

        Returns:
            Tensor: The new styles.
            Tensor: The attribute probabilities.
        """
        trajectory, _ = model(inputs, targets)
        new_styles = trajectory[-1]

        new_images = G(new_styles).clamp(-1, 1)
        outs = torch.sigmoid(image_classifier(new_images)[0])

        return new_styles, outs

    def transform(self, inputs, sources):
        """
        Transform the latent space based on the given inputs and sources.

        Args:
            inputs (Tensor): The input latent space.
            sources (Tensor): The source attributes.

        Returns:
            Tensor: The transformed latent space.
        """
        bs = inputs.size(0)
        targets = sources.clone()
        targets[:, self.attr_num] = 1 - (targets[:, self.attr_num] > 0.5).float()

        # Extract the relevant changes and apply the necessary transformation
        targets_ = targets[:, change_indices].clone()
        targets_[targets_ == 0] = -1.

        new_styles_t, outs_t = self._process_model(self.model_t, inputs, targets_)
        new_styles_s, outs_s = self._process_model(self.model_s, inputs, targets_)

        # # Combine attr_num with retained_attr_indices
        # indices = [self.attr_num] + self.retained_attr_indices

        # Combine change and keep indices
        indices = torch.cat([change_indices, keep_indices])

        kl_div_t = kl_divergence(outs_t[:, indices], targets[:, indices])
        kl_div_s = kl_divergence(outs_s[:, indices], targets[:, indices])

        bool_tensor = kl_div_t < kl_div_s
        # Expand the boolean tensor to match the shape of t1 and t2
        expanded_bool_tensor = bool_tensor.unsqueeze(1).unsqueeze(2).expand(bs, G.n_styles, G.style_dim)

        # Use the boolean tensor as an index to select the corresponding tensor where True
        new_styles = torch.where(expanded_bool_tensor, new_styles_t, new_styles_s)

        return new_styles


# ================= Dataset and DataLoader ==================== #

def create_dataloader(latents_adjusted, preds_adjusted, batch_size, num_workers):
    """
    Encapsulates the process of creating a DataLoader.
    """
    dataset = TensorDataset(latents_adjusted, preds_adjusted)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader


print('Loading dataset...')

# Load dataset
mem_ffhq_loader = create_dataloader(all_latents, all_preds, opts.batch_size, opts.num_workers)

print('Dataset loaded successfully.')

print('Processing and transforming latents...')


def process_and_transform_latents(transformer):
    """
    Process the images and transform the latent space using the provided transformer.

    Args:
        transformer: Instance of the transformer class.

    Returns:
        Concatenated tensor of all new styles.
    """
    all_new_styles = []

    for latents, sources in tqdm.tqdm(mem_ffhq_loader):
        # Move tensors to the appropriate device
        latents, sources = latents.cuda(), sources.cuda()

        # Transform the latent space
        new_styles = transformer.transform(latents, sources)
        all_new_styles.append(new_styles)

    # Concatenate all new styles
    all_new_styles = torch.cat(all_new_styles, dim=0)
    return all_new_styles


# Compute the dimension of the conditioning vector (c_dim)
c_dim = sum(1 for _ in opts.changes)

# Define the model type as DualAdaTransEditor
model_type = DualAdaTransEditor


def create_indices(attrs):
    """
    Create indices tensor

    Args:
    attrs (list): List of indices. If an element is -1, replace it with the range 40-45.

    Returns:
    torch.Tensor: Tensor containing the generated indices.
    """
    indices = []
    for attr in attrs:
        if attr != -1:
            indices.append(attr)
        else:
            indices.extend(range(40, 46))
    indices_tensor = torch.tensor(indices).long().cuda()
    return indices_tensor


# Generate an array of values from 1 to opts.max_steps (inclusive)
scales = np.arange(1, opts.max_steps + 1)

# Create indices for changes and keeps
change_indices, keep_indices = create_indices(opts.changes), create_indices(opts.keeps)

# Define paths to checkpoint files
checkpoint_paths = [
    f'/root/autodl-tmp/SL-KD/training/dual_trans_ckpts/{opts.run_name}/save_models/model_a-latest',
    f'/root/autodl-tmp/SL-KD/training/dual_trans_ckpts/{opts.run_name}/save_models/model_b-latest'
]

# Initialize a dictionary to store all time-step data for each attribute
all_attrs_steps_styles = {}

# Iterate over each attribute in the opts.changes list
for _, attr_num in enumerate(opts.changes):
    all_steps_styles = []  # Initialize a list to store all time-step data for the current attribute

    for scale in scales:
        print(f'Processing scale {scale}/{len(scales)} for attribute {attr_num}...')

        # Initialize the transformer model
        transformer = model_type(
            checkpoints=checkpoint_paths,
            attr_num=attr_num,
            scale=scale
        )

        # Process and transform latents
        step_new_styles = process_and_transform_latents(transformer)
        step_new_styles_cpu = step_new_styles.cpu()  # 将结果移到CPU
        all_steps_styles.append(step_new_styles_cpu)

        # Clear CUDA cache
        del transformer
        torch.cuda.empty_cache()

    # Stack all time-step data for the current attribute into a single tensor
    all_steps_styles_tensor = torch.stack(all_steps_styles, dim=0)

    # Store the stacked tensor in the dictionary, using the attribute number as the key
    all_attrs_steps_styles[attr_num] = all_steps_styles_tensor

# Construct string representations of changed and retained attribute indices
changed_attr_str = '_'.join(map(str, opts.changes))
retained_attr_str = '_'.join(map(str, opts.keeps))

# Define the save path, following PEP8 line-breaking conventions
save_path = (
    '/root/autodl-tmp/SL-KD/data/teacher_network_output_ckpts/'
    f'all_attrs_steps_styles_changed_{changed_attr_str}_retained_'
    f'{retained_attr_str}.pth'
)

# Save the dictionary containing all attributes and steps styles to the specified path
torch.save(all_attrs_steps_styles, save_path)
print(f'Saved all attributes\' steps styles to: {save_path}.')
