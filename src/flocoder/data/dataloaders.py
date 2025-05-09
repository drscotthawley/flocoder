from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from .datasets import PairDataset, ImageListDataset, fast_scandir, MIDIImageDataset
import random



class RandomRoll:
    """A Transform/Augmentation: 
     Randomly shifts the image in vertical direction (used for data augmentation: musical transposition)."""
    def __init__(self, max_h_shift=None, max_v_shift=2*12, p=0.5): # 2*12 means +/- 2 octaves
        self.max_h_shift = max_h_shift
        self.max_v_shift = max_v_shift
        self.p = p

    def __call__(self, img):
        import random
        if random.random() > self.p: return img
        w, h = img.size
        max_h = self.max_h_shift if self.max_h_shift is not None else w // 2
        max_v = self.max_v_shift if self.max_v_shift is not None else h // 2
        h_shift = random.randint(-max_h, max_h)
        v_shift = random.randint(-max_v, max_v)
        return img.rotate(0, translate=(h_shift, v_shift))

    def __repr__(self):
        return f"RandomRoll(max_h_shift={self.max_h_shift}, max_v_shift={self.max_v_shift}, p={self.p})"


def midi_transforms(image_size=128, random_roll=True):
    """Standard image transformations for training and validation."""
    transform_list = [
        RandomRoll() if random_roll else None,
        transforms.RandomCrop(image_size),
        transforms.ToTensor()]
    return transforms.Compose([t for t in transform_list if t is not None])


def image_transforms(image_size=128):
    return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # normalization as per ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_image_loaders(batch_size=32, image_size=128, shuffle_val=True, data_path=None, 
                         is_midi=True, num_workers=8, val_ratio=0.1, debug=True):
    is_midi = is_midi or 'pop909' in data_path.lower() or 'midi' in data_path.lower()

    # define transforms
    if is_midi: # midi piano roll images
        train_transforms = midi_transforms(image_size)
        val_transforms = midi_transforms(image_size, random_roll=False)
    else: # for regular images, e.g. from Oxford Flowers dataset
        train_transforms = image_transforms(image_size)
        val_transforms = image_transforms(image_size)
    
    if data_path is None: # fall back to Oxford Flowers dataset
        train_base = datasets.Flowers102(root='./data', split='train', transform=train_transforms, download=True)
        val_base = datasets.Flowers102(root='./data', split='val', transform=val_transforms, download=True)
    elif is_midi:
        train_base = MIDIImageDataset(split='train', transform=train_transforms, download=True, val_ratio=val_ratio)
        val_base = MIDIImageDataset(split='val', transform=val_transforms, download=True, val_ratio=val_ratio)
    else:
        # Custom directory handling, e.g. for custom datasets,...
        _, all_files = fast_scandir(data_path, ['jpg', 'jpeg', 'png'])
        if debug: 
            print(f"Found {len(all_files)} images in {data_path}")
        random.shuffle(all_files)  # Randomize order
        
        # Split into train/val (90/10 split)
        split_idx = int(len(all_files) * val_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        train_base = ImageListDataset(train_files, train_transforms)
        val_base = ImageListDataset(val_files, val_transforms)
        
    train_dataset = PairDataset(train_base)
    val_dataset = PairDataset(val_base)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)
    
    return train_loader, val_loader