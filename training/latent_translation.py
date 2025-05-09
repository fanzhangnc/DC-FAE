# Standard library imports
import argparse
import os
import sys
from itertools import chain

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F  # PyTorch functions
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# Add custom module paths to the system path
sys.path.insert(0, '/root/autodl-tmp/DC-FAE')
sys.path.insert(0, '/root/autodl-tmp/DC-FAE/models/stylegan2')

# Models operations
from models.ops import convert_to_cuda, load_network
from models.ops.grad_scaler import NativeScalerWithGradNormCount
from models.ops.loggerx import LoggerX
from models.ops.loss import IDLoss
from models.ops.lpips import LPIPS

# Models components
from models.decoder import StyleGANDecoder
from models.modules import ANT, Classifier, FLOW

# Set PyTorch print options
torch.set_printoptions(precision=3, sci_mode=False)
torch.autograd.set_grad_enabled(False)

# ================= Arguments ================ #

parser = argparse.ArgumentParser(description='Training configuration')

# Learning rates
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the transformation network')

# Training control
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID to use')
parser.add_argument('--max_iter', type=int, default=10000, help='Maximum number of iterations (batch steps)')
parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--resume', '-r', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--save_freq', type=int, default=100, help='Save frequency')

# Optimization configuration
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')

# Loss functions
parser.add_argument(
    '--cls_loss_weight', type=float, default=1.0,
    help='Weight for classification loss (Lmi), default is 1.0.'
)
parser.add_argument('--id_loss_weight', type=float, default=0.1, help='Weight for identity loss, default is 0.1.')
parser.add_argument(
    '--lpips_loss_weight', type=float, default=0.1,
    help='Weight for perceptual loss (LPIPS), default is 0.1.'
)
parser.add_argument(
    '--nll_loss_weight', type=float, default=1.0,
    help='Weight for negative log-likelihood loss (Lreg), default is 1.0.'
)
parser.add_argument('--pix_loss_weight', type=float, default=0.0, help='Weight for pixel-wise loss, default is 0.0.')
parser.add_argument(
    '--recon_loss_weight', type=float, default=1.0,
    help='Weight for reconstruction loss (Ldist), default is 1.0.'
)

# Sampling and Data Adjustment
parser.add_argument(
    '--ratio', type=float, default=1.0,
    help='Specifies the desired ratio of the number of samples in the minority '
         'class to the majority class after resampling.'
)

# WandB configuration
parser.add_argument('--entity', type=str, default='fzhang', help='Weights & Biases entity (project owner)')
parser.add_argument('--project_name', type=str, default='FaceEditing', help='Weights & Biases project name')
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')

# Experiment configuration
parser.add_argument('--run_name', default='experiment_01', type=str, help='Name of the experiment')

# Transformer configuration
parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')

# Curriculum learning settings
parser.add_argument(
    '--curriculum', type=int, default=15,
    help='Number of initial iterations with a higher emphasis on the original loss.'
)
parser.add_argument(
    '--starting_rate', type=float, default=0.01,
    help='Initial weight for balancing between the original loss and cycle loss.'
)
parser.add_argument(
    '--default_rate', type=float, default=0.5,
    help='Weight for balancing between the original loss and cycle loss after the curriculum phase.'
)

# Miscellaneous
parser.add_argument('--changes', nargs='+', type=int, default=[15], help='List of change values')
parser.add_argument('--keeps', nargs='+', type=int, default=[20, -1], help='List of keep values')
parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps for training process')

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= Paths ==================== #

# Paths to various checkpoint files
classifier_checkpoint = '/root/autodl-tmp/DC-FAE/data/focal_loss_r34_age_8410.pth'
latents_checkpoint = '/root/autodl-tmp/DC-FAE/data/ffhq_train_latents.pth'
preds_checkpoint = '/root/autodl-tmp/DC-FAE/data/focal_loss_ffhq_train_preds.pth'
realnvp_checkpoint = '/root/autodl-tmp/DC-FAE/data/realnvp.pth'
stylegan2_checkpoint_path = '/root/autodl-tmp/DC-FAE/data/ffhq.pkl'

# Initialize the StyleGAN2 decoder
output_size = 256
G = StyleGANDecoder(stylegan2_checkpoint_path, False, output_size)

# Load latent vectors and add the average latent vector
all_latents = torch.load(latents_checkpoint, map_location='cpu') + G.latent_avg

# Move the model to GPU and set to evaluation mode
G = G.to(device).eval()

# Initialize and load the classifier
image_classifier = Classifier().to(device)
state_dict = torch.load(classifier_checkpoint, map_location='cpu')
image_classifier.load_state_dict(load_network(state_dict))
image_classifier = image_classifier.eval().to(device)

# Initialize loss functions and move them to the GPU
LIPIPS_LOSS = LPIPS().to(device)
ID_LOSS = IDLoss(crop=True, backbone='r34').to(device)

# Load predictions
all_preds = torch.load(preds_checkpoint, map_location='cpu')

# Initialize and load the RealNVP model
realnvp = FLOW(
    style_dim=G.style_dim,
    n_styles=G.n_styles,
    n_layer=10
).to(device).eval()
realnvp.load_state_dict(torch.load(realnvp_checkpoint, map_location='cpu'))


# ================= Sampling and Adjustments ==================== #


def get_sampling_strategy(min_cnt, maj_cnt, ratio, flag):
    """Determine the sampling strategy based on class counts and the desired ratio."""
    epsilon = 1e-8  # Small value to prevent division by zero

    if min_cnt >= maj_cnt * ratio:
        return {flag: int(maj_cnt * ratio), 1 - flag: int(maj_cnt)}
    else:
        return {flag: int(min_cnt), 1 - flag: int(min_cnt // (ratio + epsilon))}


def get_sample_indices(labels, sampling_strategy):
    """Get the indices of the samples after applying the RandomUnderSampler."""
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

    # Fit and resample using RandomUnderSampler
    rus.fit_resample(np.random.rand(labels.size(0), 1), labels)

    return rus.sample_indices_


def adjust_sample_sizes(attribute_idx):
    """
    Adjust sample sizes based on a specified ratio and attribute index.

    Args:
        attribute_idx (int): Index of the attribute in the predictions tensor.

    Returns:
        tuple: Tensors containing (latents_a, preds_a, latents_b, preds_b).
    """
    ratio = opts.ratio

    # Extract the specified attribute's labels and convert them to binary (0/1)
    labels = (all_preds[:, attribute_idx] > 0.5).float()

    # Calculate the number of positive and negative samples
    positive_cnt = labels.sum().item()
    negative_cnt = labels.size(0) - positive_cnt

    # First sampling strategy and results
    sampling_strategy_a = get_sampling_strategy(positive_cnt, negative_cnt, ratio, 1)
    indices_a = get_sample_indices(labels, sampling_strategy_a)

    # Second sampling strategy and results
    sampling_strategy_b = get_sampling_strategy(negative_cnt, positive_cnt, ratio, 0)
    indices_b = get_sample_indices(labels, sampling_strategy_b)

    return (
        all_latents[indices_a], all_preds[indices_a],
        all_latents[indices_b], all_preds[indices_b]
    )


# ================= Dataset and DataLoader ==================== #

def create_dataloader(latents_adjusted, preds_adjusted, batch_size, num_workers):
    """
    Encapsulates the process of creating a DataLoader.
    """
    dataset = TensorDataset(latents_adjusted, preds_adjusted)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader


def update_data_and_loaders(batch_size, attribute_idx):
    """
    Adjust sample sizes and update data loaders.

    Returns:
        tuple: Contains data loaders for datasets A and B.
    """
    # Adjust sample sizes
    latents_a_adj, preds_a_adj, latents_b_adj, preds_b_adj = adjust_sample_sizes(attribute_idx)

    # Create data loaders
    loader_a = create_dataloader(
        latents_a_adj, preds_a_adj, batch_size, opts.num_workers
    )
    loader_b = create_dataloader(
        latents_b_adj, preds_b_adj, batch_size, opts.num_workers
    )

    return loader_a, loader_b


def prepare_data_loaders():
    """
    Prepare data loaders and their iterators based on pre-defined configuration.

    Returns:
        tuple: Four lists containing loaders_a, loaders_b, loader_a_iters, loader_b_iters.
    """
    # Notify that data preparation is starting
    print('==> Preparing data..')

    # Initialize lists to store loaders and iterators for each change
    loaders_a, loaders_b = [], []
    loader_a_iters, loader_b_iters = [], []

    # Calculate the total number of samples and determine the number of segments
    n_samples = opts.batch_size
    n_segments = len(opts.changes)
    seg_size = n_samples // n_segments

    # Iterate over each change specified in opts.changes
    for i, change in enumerate(opts.changes):
        # Determine batch size for the current segment
        adjusted_batch_size = seg_size if i < n_segments - 1 else n_samples - seg_size * (n_segments - 1)

        # Update data and create corresponding loaders for each change
        loader_a, loader_b = update_data_and_loaders(adjusted_batch_size, change)

        loaders_a.append(loader_a)
        loaders_b.append(loader_b)

        # Create iterators for each data loader
        loader_a_iters.append(iter(loader_a))
        loader_b_iters.append(iter(loader_b))

    return loaders_a, loaders_b, loader_a_iters, loader_b_iters


loaders_a, loaders_b, loader_a_iters, loader_b_iters = prepare_data_loaders()

# ================= Model Setup =================

# Compute the dimension of the conditioning vector (c_dim)
c_dim = sum(1 for _ in opts.changes)

# Initialize two instances of the model with specified parameters
model_a = ANT(
    c_dim=c_dim,
    hid_dim=512,
    max_steps=opts.max_steps,
    n_layers=10,
    n_styles=G.n_styles,
    style_dim=G.style_dim,
    transformer_params=transformer_params
).to(device)

model_b = ANT(
    c_dim=c_dim,
    hid_dim=512,
    max_steps=opts.max_steps,
    n_layers=10,
    n_styles=G.n_styles,
    style_dim=G.style_dim,
    transformer_params=transformer_params
).to(device)

# Combine parameters of both models for optimization
combined_params = chain(model_a.parameters(), model_b.parameters())

# Setup logger for monitoring and checkpointing
logger = LoggerX(
    save_root=os.path.join('./dual_trans_ckpts', opts.run_name),  # Directory for saving checkpoints
    config=opts,  # Configuration options
    enable_wandb=opts.wandb,  # Enable Weights and Biases logging if specified
    entity=opts.entity,  # Entity name for Weights and Biases
    name=opts.run_name,  # Run name for Weights and Biases
    project=opts.project_name  # Project name for Weights and Biases
)

# Register the models with the logger
logger.modules = [model_a, model_b]

# Resume training from the latest checkpoint if specified
if opts.resume:
    print('==> Resuming training from checkpoint..')
    logger.load_checkpoints('latest')

# ==================== Optimizer Setup ==================== #

# Initialize AdamW optimizer with specified parameters
optimizer = torch.optim.AdamW(combined_params, lr=opts.lr, weight_decay=opts.weight_decay)

# Number of gradient accumulation steps
acc_grad_steps = 8

# Initialize scaler for gradient normalization and counting
scaler = NativeScalerWithGradNormCount()


# ================= Training and Testing ================ #

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
    indices_tensor = torch.tensor(indices).long().to(device)
    return indices_tensor


def binarize_segment(tensor, mask, attr_idx):
    """Binarize a segment of the tensor based on the specified attribute index."""
    tensor[mask, attr_idx] = (tensor[mask, attr_idx] > 0.5).float()


def invert_segment(tensor, mask, attr_idx):
    """Invert the binary values in the specified segment of the tensor."""
    tensor[mask, attr_idx] = 1 - (tensor[mask, attr_idx] > 0.5).float()


def process_segments(tensor, process_fn, indices: torch.Tensor):
    """
    Apply a function to segments of the tensor based on specified indices.
    """
    n_samples = tensor.size(0)
    n_segments = change_indices.size(0)
    seg_size = n_samples // n_segments

    for i in range(n_segments):
        start_idx = i * seg_size
        end_idx = (i + 1) * seg_size if i < n_segments - 1 else n_samples

        attr_idx = change_indices[i].item()
        mask = (indices >= start_idx) & (indices < end_idx)

        process_fn(tensor, mask, attr_idx)

    return tensor


def process_sources(preds: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Process and update the predictions tensor based on segment indices and column changes.
    """
    sources = preds.clone()

    # Process segments using the provided function
    sources = process_segments(sources, binarize_segment, indices)

    # Extract columns specified by change_indices
    sources_ = sources[:, change_indices]

    # Replace zero values with -1.0
    sources_[sources_ == 0] = -1.0

    return sources_


def process_targets(preds: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process the predictions tensor and update values based on specified columns and segments.
    """
    # Clone preds to avoid modifying the original tensor
    targets = preds.clone()

    # Process segments using the invert_segment function
    targets = process_segments(targets, invert_segment, indices)

    # Extract and clone the specified columns
    targets_ = targets[:, change_indices].clone()

    # Replace zero values with -1.0
    targets_[targets_ == 0] = -1.0

    return targets, targets_


@torch.no_grad()
def test(n_iter):
    num_tests = 8
    start_index = 48
    indices = torch.arange(num_tests) + start_index

    # Extract the corresponding latents and predictions
    latents = all_latents[indices]
    preds = all_preds[indices]

    # Move the latents and preds tensors to the GPU
    latents = latents.to(device)
    preds = preds.to(device)

    # Set models to evaluation mode
    model_a.eval()
    model_b.eval()

    # Generate the original images
    out_images = [G(latents)]

    # Iterate over each attribute in opts.changes
    for attr_num in opts.changes:
        # Clone predictions and modify the specific attribute
        targets = preds.clone()
        targets[:, attr_num] = 1 - (targets[:, attr_num] > 0.5).float()

        # Extract the relevant changes and apply the necessary transformation
        targets_ = targets[:, change_indices]
        targets_[targets_ == 0] = -1.

        # Generate and append the output image from model A
        tj_a, _ = model_a(latents, targets_)

        out_image_a = G(tj_a[-1])
        out_images.append(out_image_a)

        # Generate and append the output image from model B
        tj_b, _ = model_b(latents, targets_)

        out_image_b = G(tj_b[-1])
        out_images.append(out_image_b)

    # Stack images and rearrange dimensions for grid layout
    grid_img = torch.stack(out_images)  # Shape: (num_images, C, H, W)
    grid_img = grid_img.transpose(1, 0)  # Shape: (C, num_images, H, W)
    grid_img = grid_img.reshape(-1, 3, 256, 256)  # Shape: (num_images * C, 3, 256, 256)

    # Normalize images to [0, 1]
    grid_img = (grid_img.clamp(-1, 1) + 1) * 0.5  # Convert from [-1, 1] to [0, 1]

    # Convert to PIL image and save
    grid_img_pil = to_pil_image(make_grid(grid_img, nrow=len(out_images)))
    logger.save_image(grid_img_pil, n_iter, sample_type='train')


acc_grd_step = 8
change_indices, keep_indices = create_indices(opts.changes), create_indices(opts.keeps)
syn_bs = 16

for n_iter in range(1, 1 + opts.max_iter):
    # Initialize lists to store batched latents and predictions
    latents_a_list, preds_a_list = [], []
    latents_b_list, preds_b_list = [], []

    for change_idx, _ in enumerate(opts.changes):
        try:
            # Get data from the current attribute's loader
            latents_a, preds_a = convert_to_cuda(next(loader_a_iters[change_idx]))
            latents_b, preds_b = convert_to_cuda(next(loader_b_iters[change_idx]))
        except StopIteration:
            # If any loader is exhausted, restart both iterators
            loader_a_iters[change_idx] = iter(loaders_a[change_idx])
            loader_b_iters[change_idx] = iter(loaders_b[change_idx])

            # Retrieve data again after restarting the iterators
            latents_a, preds_a = convert_to_cuda(next(loader_a_iters[change_idx]))
            latents_b, preds_b = convert_to_cuda(next(loader_b_iters[change_idx]))

        # Append the data to the corresponding lists
        latents_a_list.append(latents_a)
        preds_a_list.append(preds_a)
        latents_b_list.append(latents_b)
        preds_b_list.append(preds_b)

    with torch.autograd.set_grad_enabled(True):
        # Concatenate all batches into final tensors
        latents_a = torch.cat(latents_a_list)
        latents_b = torch.cat(latents_b_list)
        preds_a = torch.cat(preds_a_list)
        preds_b = torch.cat(preds_b_list)

        # Get batch size
        batch_size = latents_a.size(0)

        # Randomly permute indices
        indices = torch.randperm(batch_size)

        # Reorder tensors based on permuted indices
        latents_a = latents_a[indices]
        latents_b = latents_b[indices]
        preds_a = preds_a[indices]
        preds_b = preds_b[indices]

        # Set models to training mode
        model_a.train()
        model_b.train()

        # Process sources and targets
        sources_a_ = process_sources(preds_a, indices)
        sources_b_ = process_sources(preds_b, indices)
        targets_a, targets_a_ = process_targets(preds_a, indices)
        targets_b, targets_b_ = process_targets(preds_b, indices)

        # Generate styles and compute step sizes
        tj_ab, step_siz_ab = model_b(latents_a, targets_a_)
        tj_ba, step_siz_ba = model_a(latents_b, targets_b_)

        styles_ab = tj_ab[-1].float()
        styles_ba = tj_ba[-1].float()

        step_siz_ab = step_siz_ab.sum(dim=1).mean()
        step_siz_ba = step_siz_ba.sum(dim=1).mean()

        norm_ab = (styles_ab - latents_a).norm(dim=-1).mean()
        norm_ba = (styles_ba - latents_b).norm(dim=-1).mean()

        # Compute cycle loss
        tj_aba, step_siz_aba = model_a(styles_ab, sources_a_)
        tj_bab, step_siz_bab = model_b(styles_ba, sources_b_)

        styles_aba = tj_aba[-1].float()
        styles_bab = tj_bab[-1].float()

        cycle_loss_a = F.mse_loss(styles_aba, latents_a)
        cycle_loss_b = F.mse_loss(styles_bab, latents_b)

        # Generate new images from styles
        new_images_ab = G(styles_ab[:syn_bs])
        new_images_ba = G(styles_ba[:syn_bs])

        # Evaluate image classifier
        image_classifier.eval()
        outs_ab = image_classifier(new_images_ab)[0]
        outs_ba = image_classifier(new_images_ba)[0]

        # Combine change and keep indices
        class_indices = torch.cat([change_indices, keep_indices])

        # Compute classification losses
        cls_loss_b = F.binary_cross_entropy_with_logits(
            outs_ab[:, class_indices], targets_a[:syn_bs, class_indices]
        )
        cls_loss_a = F.binary_cross_entropy_with_logits(
            outs_ba[:, class_indices], targets_b[:syn_bs, class_indices]
        )

        # Generate target images with no gradient
        with torch.no_grad():
            target_images_a = G(latents_a[:syn_bs])
            target_images_b = G(latents_b[:syn_bs])

        # Compute identity losses
        id_loss_b = ID_LOSS(target_images_a, new_images_ab)
        id_loss_a = ID_LOSS(target_images_b, new_images_ba)

        # Compute perceptual losses (LPIPS)
        lpips_loss_b = LIPIPS_LOSS(target_images_a, new_images_ab)
        lpips_loss_a = LIPIPS_LOSS(target_images_b, new_images_ba)

        # Compute pixel-wise losses
        pix_loss_b = F.mse_loss(target_images_a, new_images_ab)
        pix_loss_a = F.mse_loss(target_images_b, new_images_ba)

        # Compute negative log-likelihood (NLL) losses
        nll_loss_b, logz_b, _, _ = realnvp(tj_ab[1:].view(-1, styles_ab.size(1), styles_ab.size(2)))
        nll_loss_a, logz_a, _, _ = realnvp(tj_ba[1:].view(-1, styles_ba.size(1), styles_ba.size(2)))

        # Compute reconstruction losses
        recon_loss_b = F.mse_loss(latents_a, styles_ab)
        recon_loss_a = F.mse_loss(latents_b, styles_ba)

        # Total Loss
        if n_iter < opts.curriculum:
            rate = opts.starting_rate
        else:
            rate = opts.default_rate

        # Calculate weighted sum of losses, sorted alphabetically by variable name
        weighted_loss_b = (
                cls_loss_b * opts.cls_loss_weight +
                id_loss_b * opts.id_loss_weight +
                lpips_loss_b * opts.lpips_loss_weight +
                nll_loss_b * opts.nll_loss_weight +
                pix_loss_b * opts.pix_loss_weight +
                recon_loss_b * opts.recon_loss_weight
        )

        # Calculate the final total loss for A
        total_loss_a = weighted_loss_b * (1.0 - rate) + cycle_loss_a * rate

        # Calculate weighted sum of losses, sorted alphabetically by variable name
        weighted_loss_a = (
                cls_loss_a * opts.cls_loss_weight +
                id_loss_a * opts.id_loss_weight +
                lpips_loss_a * opts.lpips_loss_weight +
                nll_loss_a * opts.nll_loss_weight +
                pix_loss_a * opts.pix_loss_weight +
                recon_loss_a * opts.recon_loss_weight
        )

        # Calculate the final total loss for B
        total_loss_b = weighted_loss_a * (1.0 - rate) + cycle_loss_b * rate

        loss = total_loss_a + total_loss_b

        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        scaler(loss, optimizer=optimizer, update_grad=update_params)

        # Log losses and related metrics, sorted alphabetically by variable name
        logger.msg([
            cls_loss_b,  # Classification loss
            id_loss_b,  # Identity loss
            lpips_loss_b,  # Perceptual loss
            logz_b,  # Log Z metric
            nll_loss_b,  # Negative log-likelihood loss
            norm_ab,  # Norm metric
            pix_loss_b,  # Pixel-wise loss
            recon_loss_b,  # Reconstruction loss
            step_siz_ab  # Step size metric
        ], n_iter)

        # Log losses and related metrics, sorted alphabetically by variable name
        logger.msg([
            cls_loss_a,  # Classification loss
            id_loss_a,  # Identity loss
            lpips_loss_a,  # Perceptual loss
            logz_a,  # Log Z metric
            nll_loss_a,  # Negative log-likelihood loss
            norm_ba,  # Norm metric
            pix_loss_a,  # Pixel-wise loss
            recon_loss_a,  # Reconstruction loss
            step_siz_ba  # Step size metric
        ], n_iter)

        if n_iter % opts.save_freq == 0:
            test(n_iter)
            logger.checkpoints('latest')
