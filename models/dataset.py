# Standard library imports
import csv
import os.path as osp
import pathlib
from functools import partial

# Third-party imports
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader


# Create FFHQ aging dataset image list
def create_FFHQ_AGING(data_root, label_file, age_group):
    # Define age clusters based on the given age group
    if age_group == 7:
        age_clusters = {'0-2': 0, '3-6': 0, '7-9': 0, '10-14': 1, '15-19': 1,
                        '20-29': 2, '30-39': 3, '40-49': 4, '50-69': 5, '70-120': 6}
    elif age_group == 10:
        age_clusters = {'0-2': 0, '3-6': 1, '7-9': 2, '10-14': 3, '15-19': 4,
                        '20-29': 5, '30-39': 6, '40-49': 7, '50-69': 8, '70-120': 9}
    else:
        raise NotImplementedError

    # Read the CSV label file
    with open(label_file, 'r', newline='') as f:
        reader = csv.DictReader(f)

        # Initialize lists for valid images and missing counts
        img_list = []
        missing_counts = 0

        # Iterate over each row in the CSV
        for csv_row in reader:
            # Extract attributes from CSV row
            age, age_conf = csv_row['age_group'], float(csv_row['age_group_confidence'])
            gender, gender_conf = csv_row['gender'], float(csv_row['gender_confidence'])
            head_pitch, head_roll, head_yaw = float(csv_row['head_pitch']), float(csv_row['head_roll']), float(
                csv_row['head_yaw'])
            left_eye_occluded, right_eye_occluded = float(csv_row['left_eye_occluded']), float(
                csv_row['right_eye_occluded'])
            glasses = csv_row['glasses']

            # Check if attributes are missing
            no_attributes_found = head_pitch == -1 and head_roll == -1 and head_yaw == -1 and \
                                  left_eye_occluded == -1 and right_eye_occluded == -1 and glasses == -1

            # Check if age and gender are reliable
            age_cond = age_conf > 0.6
            gender_cond = gender_conf > 0.66

            # Check head pose and eye occlusion conditions
            head_pose_cond = abs(head_pitch) < 30.0 and abs(head_yaw) < 40.0
            eyes_cond = (left_eye_occluded < 90.0 and right_eye_occluded < 50.0) or (
                    left_eye_occluded < 50.0 and right_eye_occluded < 90.0)
            glasses_cond = glasses != 'Dark'

            # Define valid conditions
            valid1 = age_cond and gender_cond and no_attributes_found
            valid2 = age_cond and gender_cond and head_pose_cond and eyes_cond and glasses_cond

            # Include the image if valid
            if valid1 or valid2:
                num = int(csv_row['image_number'])
                train = 1 if num < 69000 else 0
                img_filename = str(num).zfill(5) + '.png'
                glasses = 0 if glasses == 'None' else 1
                gender = 1 if gender == 'male' else 0
                age = age_clusters[age]
                img_path = osp.join(data_root, img_filename)

                # Check if the image file exists
                if not osp.exists(img_path):
                    missing_counts += 1
                    print(img_path)
                    continue

                # Add the valid image info to the list
                img_list.append([img_path, glasses, age, gender, train])

        # Print if any images are missing
        if missing_counts > 0:
            print(f'{missing_counts} images do not exist.')

    return np.array(img_list)


# Standardize the image with mean and std
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)


def create_FACE_Transform(mode, img_size):
    transform = [
        torchvision.transforms.Resize(img_size),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        normalize  # Normalize the image
    ]

    # Add random horizontal flip for training mode
    if mode == 'train':
        transform.insert(0, torchvision.transforms.RandomHorizontalFlip())

    # Return the composed transformations
    return transforms.Compose(transform)


# Dataset class for loading FFHQ age labels
class FFHQAge(torch.utils.data.Dataset):
    def __init__(self, data_root, age_group=7, img_size=128, split='train', mode='train', transform=None):
        # File path for labels
        label_file = '/root/autodl-tmp/DC-FAE/data/ffhq_aging_labels.csv'

        # Load and filter dataset using the create_FFHQ_AGING function
        self.img_list = create_FFHQ_AGING(data_root, label_file, age_group)

        # Filter data based on split
        if split == 'train':
            self.img_list = self.img_list[self.img_list[:, -1].astype(int) == 1]
        else:
            self.img_list = self.img_list[self.img_list[:, -1].astype(int) == 0]

        # Set image transformation
        self.transform = transform or create_FACE_Transform(mode, img_size)

        # Set the target as the age labels
        self.targets = torch.Tensor(self.img_list[:, 2].astype(int))
        self.age_group = age_group

    def __getitem__(self, idx):
        # Get a sample from the dataset
        line = self.img_list[idx]
        img = pil_loader(line[0])  # Load image
        if self.transform:
            img = self.transform(img)  # Apply transformation if available
        group = int(line[2].astype(int))  # Age group label
        return img, group

    def __len__(self):
        # Return total number of samples in the dataset
        return len(self.img_list)


class CelebA(torch.utils.data.Dataset):
    """
    Custom dataset for CelebA, including image loading and attribute classification.
    """

    def __init__(self, data_root, img_size=128, split='train', mode='train', transform=None, **kwargs):
        self.transform = transform if transform else create_FACE_Transform(mode, img_size)

        # Load image paths and labels
        img_list = sorted(list(pathlib.Path(data_root).rglob("*.jpg")))
        label_file = '/root/autodl-tmp/DATASET/celeba/list_attr_celeba.txt'
        eval_file = '/root/autodl-tmp/DATASET/celeba/list_eval_partition.txt'

        # Load evaluation partition
        eval_dict = {x1: int(x2) for x1, x2 in (line.strip().split(' ') for line in open(eval_file))}

        # Load attribute labels
        targets = {}
        lines = open(label_file).readlines()
        self.class2idx = lines[1].split()

        for line in lines[2:]:
            line = [l for l in line.strip().split(' ') if l != '']
            assert len(line) == 41
            targets[line[0]] = (np.array(line[1:]).astype(int) > 0).astype(int)

        # Filter images and labels based on split
        self.targets, self.img_list = [], []
        for fpath in img_list:
            fname = fpath.name
            if (
                    (split == 'train' and eval_dict[fname] != 0) or
                    (split == 'val' and eval_dict[fname] != 1) or
                    (split == 'test' and eval_dict[fname] != 2)
            ):
                continue
            self.targets.append(targets[fname])
            self.img_list.append(fpath)

        # Convert targets to tensor
        self.targets = torch.from_numpy(np.stack(self.targets)).float()

        # Select relevant classes based on class_idx
        class_idx = kwargs.get('class_idx', np.arange(40))
        self.targets = self.targets[:, class_idx]
        self.class2idx = np.array(self.class2idx)[class_idx]

    def __getitem__(self, idx):
        """
        Fetch an image and its corresponding target.
        """
        img = pil_loader(self.img_list[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.img_list)


dataset_dict = {
    'FFHQAge': FFHQAge,
    'CelebA': CelebA,
}
