# Standard library imports
import argparse
import os
import sys
import random

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import PyTorch's functional module
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

# Set system paths for custom modules
sys.path.insert(0, '/root/autodl-tmp/DC-FAE')
sys.path.insert(0, '/root/autodl-tmp/DC-FAE/models/stylegan2')

# Project-specific imports (custom modules)
from models.ops import convert_to_cuda, dataset_with_indices, load_network
from models.ops.loggerx import LoggerX
from models.ops.grad_scaler import NativeScalerWithGradNormCount
from models.ops.lpips import LPIPS
from models.ops.loss import BetweenFakeLoss, BetweenLoss, DiscriminatorFakeLoss, DiscriminatorLoss, IDLoss
from models.dataset import dataset_dict
from models.decoder import StyleGANDecoder
from models.modules import ANT, Classifier, Discriminators, FLOW

# Additional PyTorch settings
torch.autograd.set_grad_enabled(False)  # Disable autograd for specific parts of the code
torch.set_printoptions(precision=3)  # Set print precision for tensors
torch.set_printoptions(sci_mode=False)  # Disable scientific notation for printing tensors

# ================= Arguments ================ #

parser = argparse.ArgumentParser(description='Training configuration')

# Learning rates
parser.add_argument('--d_lr', default=1e-4, type=float, help='Learning rate for the discriminator')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the student network')

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
    '--cls_loss_weight',
    type=float,
    default=1.0,
    help='Weight for classification loss (Lmi), default is 1.0.'
)
parser.add_argument('--id_loss_weight', type=float, default=0.1, help='Weight for identity loss, default is 0.1.')
parser.add_argument(
    '--lpips_loss_weight',
    type=float,
    default=0.1,
    help='Weight for perceptual loss (LPIPS), default is 0.1.'
)
parser.add_argument(
    '--nll_loss_weight',
    type=float,
    default=1.0,
    help='Weight for negative log-likelihood loss (Lreg), default is 1.0.'
)
parser.add_argument('--pix_loss_weight', type=float, default=0.0, help='Weight for pixel-wise loss, default is 0.0.')

# Transformer configuration
parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')

# Loss functions
parser.add_argument('--adv', default=1, type=int, help='Flag to add discriminator or not (0 or 1)')
parser.add_argument('--btw', default=1, type=int, help='Flag to add between loss or not (0 or 1)')
parser.add_argument('--eta', default='[1] * 10', type=str, help='Eta values for discriminator loss function')
parser.add_argument('--gamma', default='[1] * 10', type=str, help='Gamma values for loss function')
parser.add_argument('--loss', default='l2', type=str, help='Loss function to use')

# WandB configuration
parser.add_argument('--entity', type=str, default='fzhang', help='Weights & Biases entity (project owner)')
parser.add_argument('--project_name', type=str, default='FaceEditing', help='Weights & Biases project name')
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')

# Experiment configuration
parser.add_argument('--run_name', default='experiment_01', type=str, help='Name of the experiment')

# Miscellaneous
parser.add_argument('--changes', nargs='+', type=int, default=[15, ], help='List of change values')
parser.add_argument(
    '--grl',
    action='store_true',  # Use `store_true` action to handle boolean values
    help='Enable gradient reversal layer'
)
parser.add_argument('--keeps', nargs='+', type=int, default=[20, -1], help='List of keep values')
parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps for training process')
parser.add_argument('--out_layer', default='[-1]', type=str, help='Indices of steps for output')

# Add p_t argument here
parser.add_argument('--p_t', type=float, nargs='+', help='List of probabilities for each attribute in opts.changes')

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

# Convert elements of opts.changes and opts.keeps to strings and join them with underscores for filenames
change_values = '_'.join(map(str, opts.changes))
keep_values = '_'.join(map(str, opts.keeps))

# Define the save path, following PEP8 line-breaking conventions
file_name = f'all_attrs_steps_styles_changed_{change_values}_retained_{keep_values}.pth'

# ================= Paths ==================== #

# Path to the StyleGAN2 checkpoint file
stylegan2_checkpoint_path = '/root/autodl-tmp/DC-FAE/data/ffhq.pkl'

# Path to the file containing latent vectors for the FFHQ training dataset
latents_checkpoint = '/root/autodl-tmp/DC-FAE/data/ffhq_train_latents.pth'

# Path to the file containing prediction results for the FFHQ training dataset
preds_checkpoint = '/root/autodl-tmp/DC-FAE/data/focal_loss_ffhq_train_preds.pth'

# Path to the checkpoints for new styles generated by steps, utilizing a dynamic file name
steps_new_styles_checkpoint = f'/root/autodl-tmp/DC-FAE/data/teacher_network_output_ckpts/{file_name}'

classifier_checkpoint = '/root/autodl-tmp/DC-FAE/data/focal_loss_r34_age_8410.pth'

realnvp_checkpoint = '/root/autodl-tmp/DC-FAE/data/realnvp.pth'

# Load model and data using the defined paths
output_size = 256

G = StyleGANDecoder(
    stylegan2_checkpoint_path,
    False,
    output_size,
)

# Move the model to GPU and set to evaluation mode
G = G.to(device).eval()

# Initialize the image classifier and move it to the CUDA device
image_classifier = Classifier().to(device)

# Load the state dictionary into the classifier
image_classifier.load_state_dict(load_network(torch.load(classifier_checkpoint, map_location='cpu')))

# Move the classifier to the CUDA device and set it to evaluation mode
image_classifier = image_classifier.eval().to(device)

# Initialize LPIPS loss and move it to the CUDA device
LIPIPS_LOSS = LPIPS().to(device)

# Initialize ID loss, move it to CUDA, and configure it with cropping and ResNet34 backbone
ID_LOSS = IDLoss(crop=True, backbone='r34').to(device)

# Load latents and move to CUDA
all_latents = torch.load(latents_checkpoint, map_location='cpu').to(device) + G.latent_avg

# Load predictions and move to CUDA
all_preds = torch.load(preds_checkpoint, map_location='cpu').to(device)

realnvp = FLOW(style_dim=G.style_dim, n_styles=G.n_styles, n_layer=10).cuda().eval()
realnvp.load_state_dict(torch.load(realnvp_checkpoint, map_location='cpu'))

# Load new styles tensor from checkpoint and move to CPU
all_steps_new_styles = torch.load(steps_new_styles_checkpoint, map_location='cpu')

# Specify the layers to move to CUDA
out_layer_indices = eval(opts.out_layer)

# Extract the specified layers from each tensor in the dictionary and move to device
selected_layers = {
    attr: tensor[out_layer_indices, :, :, :]  # Assuming out_layer_indices is for the first dimension
    for attr, tensor in all_steps_new_styles.items()
}

# ================= Transformations ==================== #

test_transform = transforms.Compose([
    # Resize image to 256x256 pixels
    transforms.Resize(256),

    # Convert image to tensor and scale pixel values from [0, 255] to [0, 1]
    transforms.ToTensor(),

    # Normalize with mean and std of 0.5 for each channel (R, G, B), in-place
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

# ================= Dataset and DataLoader ==================== #

print('==> Preparing data...')

# Set dataset path and type
data_root = '/root/autodl-tmp/DATASET/FFHQ/images256x256'
dataset_type = dataset_dict['FFHQAge']

# Create dataset instance with specified parameters
ffhq_dataset = dataset_type(data_root, img_size=256, split='train', transform=test_transform)

# Define DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset_with_indices(ffhq_dataset),
    batch_size=opts.batch_size,
    shuffle=True,  # Shuffle data to improve model generalization
    drop_last=True,  # Drop the last batch if dataset size is not divisible by batch_size
    num_workers=opts.num_workers
)

# Create an iterator for the training DataLoader
train_loader_iter = iter(train_loader)

# ================= Model Setup ================ #

# Determine the dimension of the conditioning vector (c_dim)
c_dim = sum(6 if attr_num == -1 else 1 for attr_num in opts.changes)

# Initialize the model with specified parameters and move it to the GPU
student = ANT(
    c_dim=c_dim,
    hid_dim=512,
    max_steps=opts.max_steps,
    n_layers=10,
    n_styles=G.n_styles,
    style_dim=G.style_dim,
    transformer_params=transformer_params
).to(device)

# Generate a list of dimensions, each being G.n_styles * G.style_dim, with length equal to the number of elements in
# opts.out_layer
output_dims = [G.style_dim for _ in eval(opts.out_layer)]

# Print the generated dimensions
print('Output dimensions:', output_dims)

# Initialize parameters to update
update_parameters = [{'params': student.parameters()}]

# Add discriminators if adversarial training is enabled
discs = None
if opts.adv:
    discs = Discriminators(output_dims, grl=opts.grl)
    for disc in discs.discriminators:
        disc = disc.to(device)
        update_parameters.append({'params': disc.parameters(), 'lr': opts.d_lr})

# Set up logging and the model
logger = LoggerX(
    save_root=os.path.join('./compressed_ckpt', opts.run_name),  # Directory to save checkpoints
    config=opts,  # Configuration options
    enable_wandb=opts.wandb,  # Enable Weights and Biases logging if specified
    entity=opts.entity,  # Entity name for Weights and Biases
    name=opts.run_name,  # Run name for Weights and Biases
    project=opts.project_name  # Project name for Weights and Biases
)

# Add the model `student` to the logger's modules
logger.modules = [student]

# Resume training from checkpoint if specified
if opts.resume:
    print('==> Resuming from checkpoint..')

    logger.load_checkpoints('latest')

# ==================== Loss Function for Generator ==================== #

# Define real BetweenLoss or use BetweenFakeLoss depending on the condition
if opts.btw:
    criterion = BetweenLoss(eval(opts.gamma))  # Example of using real BetweenLoss
else:
    criterion = BetweenFakeLoss()  # Use the fake loss when not needed

# ==================== Loss Function for Discriminator ==================== #

if opts.adv:
    # Use actual discriminator loss if adversarial training is enabled
    discriminator_criterion = DiscriminatorLoss(discs, eval(opts.eta))
else:
    # Use a placeholder loss function if no actual discriminator loss is needed
    discriminator_criterion = DiscriminatorFakeLoss()

# ==================== Optimizer Setup ==================== #

# Batch size for synthetic data
syn_bs = 16

# Number of gradient accumulation steps
acc_grad_steps = 8

# Initialize AdamW optimizer with specified learning rate and weight decay
optimizer = torch.optim.AdamW(update_parameters, lr=opts.lr, weight_decay=opts.weight_decay)

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


def compute_weights(targets_, p_t):
    batch_size, num_attrs = targets_.shape
    weights = torch.zeros((batch_size, 1))

    for a in range(num_attrs):
        # Select samples where attribute 'a' is either 1.0 or -1.0
        mask = (targets_[:, a] == 1.0) | (targets_[:, a] == -1.0)

        # Calculate the total number of selected samples
        total_count = mask.sum().item()

        pos_count = (targets_[:, a] == 1.0).sum().item()
        p_b_a = pos_count / total_count

        if p_b_a > p_t[a]:  # Over-represented
            pos_indices = (targets_[:, a] == 1.0).nonzero(as_tuple=True)[0]
            num_to_keep = int(p_t[a] * total_count)
            keep_indices = pos_indices[torch.randperm(len(pos_indices))[:num_to_keep]]
            weights[keep_indices] = 1

            weights[targets_[:, a] == -1.0] = (1 - p_t[a]) / (1 - p_b_a)
        else:  # Under-represented
            neg_indices = (targets_[:, a] == -1.0).nonzero(as_tuple=True)[0]
            num_to_keep = int((1 - p_t[a]) * total_count)
            keep_indices = neg_indices[torch.randperm(len(neg_indices))[:num_to_keep]]
            weights[keep_indices] = 1

            weights[targets_[:, a] == 1.0] = p_t[a] / p_b_a

    return weights


def sample_indices(targets_, p_t, syn_bs):
    """Selects balanced indices for each attribute based on class probabilities."""
    _, num_attrs = targets_.shape
    sel_indices = []

    for a in range(num_attrs):
        # Get indices for positive and negative samples
        pos_indices = (targets_[:, a] == 1.0).nonzero().squeeze()
        neg_indices = (targets_[:, a] == -1.0).nonzero().squeeze()

        # Calculate the number of positive and negative samples needed
        pos_cnt = int(p_t[a] * syn_bs)
        neg_cnt = syn_bs - pos_cnt

        # If there are fewer samples than needed, repeat the indices to balance the count
        if len(pos_indices) < pos_cnt:
            pos_indices = pos_indices.repeat((pos_cnt // len(pos_indices)) + 1)[:pos_cnt]
        if len(neg_indices) < neg_cnt:
            neg_indices = neg_indices.repeat((neg_cnt // len(neg_indices)) + 1)[:neg_cnt]

        # Randomly select the required number of positive and negative indices
        sel_pos_indices = random.sample(pos_indices.tolist(), pos_cnt)
        sel_neg_indices = random.sample(neg_indices.tolist(), neg_cnt)

        # Extend the selected indices list
        sel_indices.extend(sel_pos_indices)
        sel_indices.extend(sel_neg_indices)

    return torch.tensor(sel_indices)


def select_layers_and_construct_batch(layer_index, data_indices, trajectory, new_styles, attrs):
    filtered_trajectory = trajectory[1:]

    # Select the specified layer from the filtered trajectory
    selected_trajectory = filtered_trajectory[layer_index, :, :, :]

    # Initialize a list to hold the selected new_styles for the current batch
    selected_new_styles_list = []

    # Loop over data_indices to select new_styles based on the corresponding attributes
    for idx, data_index in enumerate(data_indices):
        attr = attrs[idx]  # Get the attribute of the current sample
        selected_style = new_styles[attr]  # Select the corresponding new_styles tensor from the dict
        selected_new_styles_list.append(selected_style[:, data_index.cpu(), :, :])  # Select and append the tensor

    # Stack the list into a new tensor along the batch dimension
    selected_new_styles = torch.stack(selected_new_styles_list, dim=1).to(device)

    return selected_trajectory, selected_new_styles


@torch.no_grad()
def test(n_iter):
    """
    Function to test the model with a given number of iterations.

    Args:
        n_iter (int): The number of iterations for the test.

    Returns:
        None
    """
    num_tests = 8
    start_index = 48
    indices = torch.arange(num_tests) + start_index

    # Create a list of image tensors by indexing the dataset for each index in 'indices'
    image_tensors = [ffhq_dataset[index][0] for index in indices]

    # Stack the list of image tensors into a single tensor
    images = torch.stack(image_tensors)

    # Move the stacked tensor to the specified device (e.g., GPU or CPU)
    images = images.to(device)

    # Extract the corresponding latents and predictions
    latents = all_latents[indices]
    sources = all_preds[indices]

    student.eval()
    output_images = [images, G(latents)]

    # Iterate over each attribute in opts.changes
    for attr_num in opts.changes:
        # Clone predictions and modify the specific attribute
        targets = sources.clone()
        targets[:, attr_num] = 1 - (targets[:, attr_num] > 0.5).float()

        # Extract the relevant changes and apply the necessary transformation
        targets_ = targets[:, change_indices]
        targets_[targets_ == 0] = -1.

        # Obtain the outputs from the student model
        outputs, _ = student(latents, targets_)

        # Extract the last tensor from the model's output
        last_output = outputs[-1]

        # Process the last output tensor with G_parallel
        processed_image = G(last_output)

        # Append the processed image to the output list
        output_images.append(processed_image)

    # Stack images and rearrange dimensions for grid layout
    grid_img = torch.stack(output_images)  # Shape: (num_images, C, H, W)
    grid_img = grid_img.transpose(1, 0)  # Shape: (C, num_images, H, W)
    grid_img = grid_img.reshape(-1, 3, 256, 256)  # Shape: (num_images * C, 3, 256, 256)

    # Normalize images to [0, 1]
    grid_img = (grid_img.clamp(-1, 1) + 1) * 0.5  # Convert from [-1, 1] to [0, 1]

    # Convert to PIL image and save
    grid_img_pil = to_pil_image(make_grid(grid_img, nrow=len(output_images)))
    logger.save_image(grid_img_pil, n_iter, sample_type='train')


# change_indices, keep_indices = create_indices(opts.changes), create_indices(opts.compressed_keep_values)
change_indices, keep_indices = create_indices(opts.changes), create_indices(opts.keeps)

# Initialize probability tensor based on opts.p_t or use default values
if opts.p_t:
    p_t = torch.tensor(opts.p_t)
    print(f'p_t tensor: {p_t}')
else:
    # Default to a tensor with 0.5 for each attribute in opts.changes
    p_t = torch.tensor([0.5] * len(opts.changes))
    print(f'No p_t provided. Using default tensor: {p_t}')

for n_iter in range(1, 1 + opts.max_iter):
    try:
        # Attempt to get the next batch from the training loader
        (images, _), indices = convert_to_cuda(next(train_loader_iter))
    except StopIteration:
        # Restart the iterator if StopIteration is raised
        train_loader_iter = iter(train_loader)
        (images, _), indices = convert_to_cuda(next(train_loader_iter))

    with torch.autograd.set_grad_enabled(True):
        # Get latents and source predictions for the current batch
        latents = all_latents[indices]
        sources = all_preds[indices]

        bs = latents.size(0)
        student.train()
        targets = sources.clone()
        attributes = []

        for i in range(bs):
            # Randomly select an attribute from the options
            attr = random.choice(opts.changes)
            attributes.append(attr)

            # Flip the target value for the selected attribute (1 -> 0 or 0 -> 1)
            targets[i, attr] = 1 - (targets[i, attr] > 0.5).float()

        # Clone the relevant targets for modification based on the change indices
        targets_ = targets[:, change_indices].clone()
        targets_[targets_ == 0] = -1.

        selected_idx = sample_indices(targets_, p_t, syn_bs)

        # Compute the weights based on the modified targets
        weights = compute_weights(targets_, p_t).to(device)

        # Concatenate weights twice along dimension 0
        concatenated_weights = torch.cat((weights, weights), dim=0)

        # Repeat each element of concatenated weights along dimension 1
        discriminator_weights = concatenated_weights.repeat(1, 2)

        # Generate trajectory and adjust styles using the student model
        trajectory, step_size = student(latents, targets_)

        # Select specific layers and construct a batch of new styles
        selected_trajectory, selected_new_styles = select_layers_and_construct_batch(
            eval(opts.out_layer), indices, trajectory, selected_layers, attributes
        )

        # Compute loss based on selected trajectory, new styles, and weights
        loss = criterion(selected_trajectory, selected_new_styles, weights)

        # Initialize BCEWithLogitsLoss with discriminator weights
        bce_loss_fn = nn.BCEWithLogitsLoss(weight=discriminator_weights)

        # Compute the discriminator loss between the selected trajectory and new styles
        d_loss = discriminator_criterion(
            selected_trajectory,
            selected_new_styles,
            bce_loss_fn
        )
        # Get the new styles from the last trajectory step
        new_styles = trajectory[-1]
        new_styles = new_styles.float()

        # Calculate step size and normalize styles
        step_size = step_size.sum(dim=1).mean()
        norm = (new_styles - latents).norm(dim=-1).mean()

        # Generate new images using the generator with the selected styles
        new_images = G(new_styles[selected_idx])

        # Set image classifier to evaluation mode and classify new images
        image_classifier.eval()
        outs = image_classifier(new_images)[0]

        # Concatenate change and keep indices for classification loss
        class_indices = torch.cat([change_indices, keep_indices])

        # Compute classification loss
        cls_loss = F.binary_cross_entropy_with_logits(
            outs[:, class_indices], targets[selected_idx][:, class_indices]
        )

        # Generate target images and compute losses without gradient tracking
        with torch.no_grad():
            target_images = G(latents[selected_idx])

        id_loss = ID_LOSS(target_images, new_images)
        lpips_loss = LIPIPS_LOSS(target_images, new_images)
        pix_loss = F.mse_loss(target_images, new_images)

        # Compute NLL loss and other outputs from realnvp model
        nll_loss, logz, _, _ = realnvp(trajectory[1:].view(-1, new_styles.size(1), new_styles.size(2)))

        # Total loss is the sum of style transfer loss and discriminator loss
        total_loss = (
                loss +
                d_loss +
                cls_loss * opts.cls_loss_weight +
                id_loss * opts.id_loss_weight +
                lpips_loss * opts.lpips_loss_weight +
                nll_loss * opts.nll_loss_weight +
                pix_loss * opts.pix_loss_weight
        )

        # Divide total loss by the number of accumulation steps for gradient update
        total_loss = total_loss / acc_grad_steps
        update_params = (n_iter % acc_grad_steps == 0)

        # Scale gradients and update parameters
        scaler(total_loss, optimizer=optimizer, update_grad=update_params)

        # Log losses
        logger.msg(
            [loss, d_loss, cls_loss, id_loss, lpips_loss, logz, nll_loss, norm, pix_loss, step_size],
            n_iter
        )

        # Save checkpoints periodically
        if n_iter % opts.save_freq == 0:
            test(n_iter)
            logger.checkpoints('latest')
