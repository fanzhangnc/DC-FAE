import torch
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Optional, List, Tuple, Union


def load_network(state_dict):
    """
    Load and process a state dictionary, removing any 'module.' prefixes.

    Args:
        state_dict (str or dict): Path to the state dictionary file or the state dictionary itself.

    Returns:
        OrderedDict: Processed state dictionary without 'module.' prefixes.
    """
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')

    # Create new OrderedDict that does not contain 'module.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # Remove 'module.'
        new_state_dict[namekey] = v

    return new_state_dict


def resize(x, s):
    """
    Resize the input tensor to the specified size using bicubic interpolation.

    Args:
        x (torch.Tensor): The input tensor to resize. Expected to have dimensions (N, C, H, W).
        s (int): The target size for height and width.

    Returns:
        torch.Tensor: The resized tensor.
    """
    if x.size(2) == s and x.size(3) == s:
        return x
    return F.interpolate(x, size=(s, s), mode='bicubic', align_corners=True, antialias=True)


def convert_to_cuda(data):
    """
    Converts each NumPy array data field into a tensor on CUDA.

    Args:
        data: The data to convert, which can be a tensor, mapping, named tuple, or sequence.

    Returns:
        The data converted to CUDA tensors if applicable.
    """
    elem_type = type(data)

    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, Mapping):
        return {key: convert_to_cuda(value) for key, value in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # If it's a namedtuple, convert each of its elements
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [convert_to_cuda(d) for d in data]
    else:
        return data

class dataset_with_indices(torch.utils.data.Dataset):
    """
    A wrapper for datasets that returns the data and its index.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to wrap.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Fetches the item and its index from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (data, index)
        """
        data = self.dataset[idx]
        return [data, idx]


def age2group(
        age: Optional[torch.Tensor] = None,
        groups: Optional[torch.Tensor] = None,
        age_group: int = 4,
        bins: Optional[Union[List, Tuple]] = None,
        ordinal: bool = False
) -> torch.Tensor:
    """
    Categorize ages into groups based on specified bins or predefined age groupings.

    Args:
        age: Tensor of ages to be grouped.
        groups: Tensor to store the groups. If None, will be created based on `age`.
        age_group: Number of age groups to divide into (default: 4).
        bins: Custom bins to use for grouping ages.
        ordinal: If True, convert groups to ordinal representation.

    Returns:
        Tensor containing the age groups.
    """

    # Initialize groups tensor if not provided
    if groups is None:
        assert age is not None, "Age tensor must be provided if groups tensor is not initialized."
        groups = torch.zeros_like(age).to(age.device)

        # Define bins if provided, else use predefined sections based on age_group
        if bins is not None:
            section = bins
            age_group = len(section) + 1
        else:
            if age_group == 4:
                section = [30, 40, 50]
            elif age_group == 5:
                section = [20, 30, 40, 50]
            elif age_group == 6:
                section = [10, 20, 30, 40, 50]
            elif age_group == 7:
                section = [10, 20, 30, 40, 50, 60]
            elif age_group == 8:
                # Age groups: 0-12, 13-18, 19-25, 26-35, 36-45, 46-55, 56-65, 66+
                section = [12, 18, 25, 35, 45, 55, 65]
            else:
                raise NotImplementedError("Age group not supported.")

        # Assign group labels based on age thresholds
        for i, thresh in enumerate(section, 1):
            groups[age > thresh] = i  # Assign group number based on the threshold

    # Handle ordinal encoding if specified
    if ordinal:
        # Convert groups to long type for one-hot encoding
        groups = groups.long()

        # Perform one-hot encoding of the groups tensor with the specified number of age groups
        ordinal_labels = F.one_hot(groups, age_group)

        # Iterate through each element in the groups tensor
        for i in range(groups.size(0)):
            # Set all elements up to the group value to 1 for ordinal encoding
            ordinal_labels[i, :groups[i]] = 1.

        # Remove the first column to convert to ordinal encoding and match the device of the age tensor
        ordinal_labels = ordinal_labels[:, 1:].to(age)

        # Update groups to be the ordinal labels
        groups = ordinal_labels

    return groups
