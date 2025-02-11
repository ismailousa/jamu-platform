import ast
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to CSV with columns 'path' and 'target' (a one-hot encoded string, e.g., "[1,0,0,0,0,0]").
            transform (callable, optional): Transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.image_dir / row['filename']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert one-hot target (string) to an integer label.
        target_list = ast.literal_eval(row['target'])
        label = target_list.index(1)
        return image, label