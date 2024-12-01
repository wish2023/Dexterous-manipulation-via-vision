from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

class RGBDGraspDataset(Dataset):
    def __init__(self, datasets):
        """
        Initialize the dataset by combining multiple sources.
        
        Args:
        datasets (list of dict): Each dict contains:
            - 'image_path': Path to RGB images.
            - 'depth_path': Path to depth maps.
            - 'anno_path': Path to JSON annotation file(s).
            - 'image_subdirs': List of subdirectories for images (optional).
        """
        self.data = []
        self.grasp_mapping = {
            "3 jaw chuck": 0,
            "key": 1,
            "pinch": 2,
            "power": 3,
            "tool": 4
            }
        
        for dataset in datasets:
            image_path = dataset['image_path']
            depth_path = dataset['depth_path']
            anno_path = dataset['anno_path']
            subdirs = dataset.get('image_subdirs', None)

            if os.path.isdir(anno_path):
                anno_files = [os.path.join(anno_path, f) for f in os.listdir(anno_path) if f.endswith('.json')]
                annotations = {}
                for file in anno_files:
                    with open(file, 'r') as f:
                        annotations.update(json.load(f))
            else:
                with open(anno_path, 'r') as f:
                    annotations = json.load(f)


            if subdirs:
                for subdir in subdirs:
                    full_image_dir = os.path.join(image_path, subdir)
                    full_depth_dir = os.path.join(depth_path, subdir)
                    for img_file in os.listdir(full_image_dir):
                        if img_file in annotations:
                            rgb_path = os.path.join(full_image_dir, img_file)
                            depth_file = os.path.splitext(img_file)[0] + "_depth.png"
                            depth_filepath = os.path.join(full_depth_dir, depth_file)

                            label = annotations[img_file]["grip"]
                            if os.path.exists(rgb_path) and os.path.exists(depth_filepath):
                                self.data.append((rgb_path, depth_filepath, label))


            else:
                for img_file in os.listdir(image_path):
                    if img_file in annotations:
                        rgb_path = os.path.join(image_path, img_file)
                        depth_file = os.path.splitext(img_file)[0] + "_depth.png"
                        depth_filepath = os.path.join(depth_path, depth_file)

                        label = annotations[img_file]["grip"]
                        if label != 'None':
                            self.data.append((rgb_path, depth_filepath, label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.data[idx]
        
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_tensor = transform(rgb_image)

        depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)[:, :, 0]
        depth_tensor = transform(depth_image)

        label = self.grasp_mapping[label]

        return rgb_tensor, depth_tensor, label

# Example usage
datasets = [
    {
        'image_path': '../data/DeepGrasping_JustImages',
        'depth_path': '../data/DEPTH_DeepGrasping_JustImages',
        'anno_path': '../data/DeepGrasping_Anno',
        'image_subdirs': [f'{i:02}' for i in range(1, 11)],
    },
    {
        'image_path': '../data/Imagenet',
        'depth_path': '../data/DEPTH_Imagenet',
        'anno_path': '../data/Anno_ImageNet.json',
    },
    {
        'image_path': '../data/HandCam',
        'depth_path': '../data/DEPTH_HandCam',
        'anno_path': '../data/Anno_HandCam4.json',
    }
]

dataset = RGBDGraspDataset(datasets)
print(f"Dataset size: {len(dataset)}")
rgb, depth, label = dataset[3000]
