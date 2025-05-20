import os
import re
from pathlib import Path
from PIL import Image
import random
import torch 
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split
from torchvision import transforms, datasets

from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
from tqdm import tqdm

from .pianoroll import midi_to_pr_img

# general utility
def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list  # list of allowed file extensions
    ):
    """very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243
       copy-pasted from github/drscotthawley/aeio/core  
    """
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in ext:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files



### Transforms

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


def image_transforms(image_size=128, 
                     means=[0.485, 0.456, 0.406], # as per common ImageNet metrics
                     stds=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        transforms.RandomRotation(degrees=15, fill=means),
        transforms.Lambda(lambda img: transforms.CenterCrop(int(min(img.size) * 0.9))(img)), # Crop to 90% to avoid rotation artifacts
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])


### end Transforms



### Datasets 

class PairDataset(Dataset):
    """This is intended to grab input,target pairs of image datasets along with class info for the images.
       But for now, it just returns the target as the same as the input 
          (e.g. for training true autoencoders in reconstruction)
       This intended for training on standard datasets like MNIST, CIFAR10, OxfordFlowers, etc.
       TODO: expand this later for more generate input/target pairs.
    """
    def __init__(self, base_dataset:Dataset, return_filenames=False):
        self.dataset, self.indices = base_dataset, list(range(len(base_dataset)))
        self.return_filenames = return_filenames

    def __len__(self): 
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Get source image and class
        source_img, source_class = self.dataset[idx]
        target_idx = idx # random.choice(self.indices) # TODO: for now just do reconstruction.
        target_img, target_class = self.dataset[target_idx]
        
        if not self.return_filenames:
            return source_img, source_class, target_img, target_class
        else:
            return source_img, source_class, target_img, target_class, self.file_list[idx], self.file_list[target_idx]


class ImageListDataset(Dataset):
    """ for custom datasets that are just lists of image files """
    def __init__(self, 
                 file_list,      # list of image file paths, e.g. from fast_scandir
                 transform=None, # can specify transforms manually, i.e. outside of dataloader. but usually we let the dataloader do transforms
                 split='all',       # 'train', 'val', or 'all'
                 val_ratio=0.1,  # percentage for validation
                 seed=42,        # for reproducibility 
                 debug=True):
        self.files = file_list
        # Apply split if needed
        if split != 'all' and len(file_list) > 0:
            random.seed(seed)  # For reproducibility
            all_files = file_list.copy()  # Make a copy to avoid modifying the original
            random.shuffle(all_files)
            split_idx = int(len(all_files) * (1 - val_ratio))
            
            if split == 'train':
                self.files = all_files[:split_idx]
            else:  # 'val'
                self.files = all_files[split_idx:]

        self.actual_len = len(self.files)
        self.images = [None]*self.actual_len
        self.transform = transform
        if debug:
            print(f"Dataset contains {self.actual_len} images")
        
    def __len__(self):
        return self.actual_len 
        
    def __getitem__(self, idx):
        actual_idx = idx % self.actual_len  # Use modulo to wrap around the index
        if self.images[actual_idx] is None: # lazy "pre-loading": it will eventualy store all images in CPU memory
            self.images[actual_idx] = Image.open(self.files[actual_idx]).convert('RGB')
        img = self.images[actual_idx]

        if self.transform:
            img = self.transform(img)
        return img, 0  
    

class MIDIImageDataset(ImageListDataset):
    """ This renders a midi dataset (POP909 by default) as images """
    def __init__(self, 
                 root=Path.home() / "datasets",  # root directory for the MIDI part of the dataset
                 url = "https://github.com/music-x-lab/POP909-Dataset/raw/refs/heads/master/POP909.zip", # url for downloading the dataset
                 transform=None, # can specify transforms manually, i.e. outside of dataloader
                 split='all',       # 'train', 'val', or 'all'
                 val_ratio=0.1,  # percentage for validation
                 seed=42,        # for reproducibility 
                 skip_versions=True, # if true, it will skip the extra versions of the same song
                 total_only=True, # if true, it will only keep the "_TOTAL_" version of each song
                 download=True, # if true, it will download the datase -- leave this on for now
                 debug=False):
        
        if download: datasets.utils.download_and_extract_archive(url, download_root=root)
        download_dir = root / url.split("/")[-1].replace(".zip", "")
        self.midi_files = fast_scandir(download_dir, ['mid', 'midi'])[1]
        if not self.midi_files or len(self.midi_files) == 0:
            raise FileNotFoundError(f"No MIDI files found in {download_dir}")
        if skip_versions: 
            self.midi_files = [f for f in self.midi_files if '/versions/' not in f]

        
        if debug: 
            print(f"download_dir: {download_dir}")
            print(f"len(midi_files): {len(self.midi_files)}")
            #print(f"midi_files: {self.midi_files}") 

        # convert midi files to images
        self.midi_img_dir = download_dir.with_name(download_dir.name + "_images")
        if debug: print(f"midi_img_dir = {self.midi_img_dir}")
        if not self.midi_img_dir.exists():
            self.midi_img_dir.mkdir(parents=True, exist_ok=True)
            self.convert_all()
        else: 
            print(f"{self.midi_img_dir} already exists, skipping conversion")

        self.midi_img_file_list = fast_scandir(self.midi_img_dir, ['.png'])[1]  # get the list of image files
        if not self.midi_img_file_list:
            raise FileNotFoundError(f"No image files found in {self.midi_img_dir}")
        if total_only:
            self.midi_img_file_list = [f for f in self.midi_img_file_list if '_TOTAL' in f]
        if debug: print(f"len(midi_img_file_list): {len(self.midi_img_file_list)}")

        super().__init__(self.midi_img_file_list, transform=transform, 
                         split=split, val_ratio=val_ratio, seed=seed, debug=debug)
 
    def convert_one(self, midi_file, debug=True):
        if debug: print(f"Converting {midi_file} to image")
        midi_to_pr_img(midi_file, self.midi_img_dir, show_chords=False, all_chords=None, 
                          chord_names=None, filter_mp=True, add_onsets=True,
                          remove_leading_silence=True)

    def convert_all(self):
        process_one = partial(self.convert_one)
        num_cpus = cpu_count()
        with Pool(num_cpus) as p:
            list(tqdm(p.imap(process_one, self.midi_files), total=len(self.midi_files), desc='Processing MIDI files'))



class InfiniteDataset(IterableDataset):
    """ This is a wrapper around a dataset that allows for infinite iteration.
        It randomly samples from the base dataset indefinitely.
        e.g. 
        base_dataset = MIDIImageDataset(transform=transform)
        dataset = InfiniteImageDataset(base_dataset)
    """
    def __init__(self, base_dataset, shuffle=True):
        super().__init__()
        self.dataset = base_dataset
        self.actual_len = len(self.dataset)
        assert shuffle, "InfiniteDataset only supports shuffle=True for now"
    
    def __iter__(self):
        while True:
            # Generate a random index
            idx = random.randint(0, self.actual_len - 1)
            # Get the item from the base dataset
            yield self.dataset[idx]


class PreEncodedDataset(Dataset):
    """for data pre-encoded by the Encoder of the VQVAE model"""
    def __init__(self, data_dir="/data/encoded-POP909", max_cache_items=10000):
        data_dir = os.path.expanduser(data_dir)
        print(f"PreEncodedDataset: searching in {data_dir}")
        self.data_dir = Path(data_dir)
        
        # Use fast_scandir for better performance instead of glob
        subdirs, files = fast_scandir(str(self.data_dir), ['pt'])
        self.files = [Path(f) for f in files]
        print(f"PreEncodedDataset: found {len(self.files)} files")
        
        # Extract class from filenames using regex
        #pattern = re.compile(r'.*_class(\d+)_.*\.pt')
        pattern = re.compile(r'sample_\d+_(\d+)_[a-z0-9]+\.pt')
        
        # Get all unique class numbers to determine total number of classes
        self.class_numbers = set()
        for f in self.files:
            match = pattern.match(f.name)
            if match:
                self.class_numbers.add(int(match.group(1)))
        
        self.n_classes = len(self.class_numbers) if self.class_numbers else 0
        #print(f"Found {len(self.files)} encoded samples across {self.n_classes} classes")
        
        # Initialize cache for faster access
        self.cache = {}
        self.max_cache_items = max_cache_items
        print(f"Using memory cache with max {max_cache_items} items")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # If not in cache, load from disk
        file_path = self.files[idx]

        # Extract class from filename if present
        #match = re.match(r'.*_class(\d+)_.*\.pt', file_path.name)
        match = re.match(r'sample_\d+_(\d+)_[a-z0-9]+\.pt', file_path.name)
        class_idx = int(match.group(1)) if match else 0
        #print(f"Loaded sample from {file_path.name} with class_idx: {class_idx}")
        
        # Load the encoded tensor directly
        encoded = torch.load(file_path, map_location='cpu')
        
        # Create the return tuple
        item = (encoded, torch.tensor(class_idx, dtype=torch.long))
        
        # Add to cache if not full
        if len(self.cache) < self.max_cache_items:
            self.cache[idx] = item
        # If cache is full, replace a random item
        # This is simpler than LRU and still effective for random access patterns
        elif random.random() < 0.01:  # 1% chance to replace an existing cache item
            random_key = random.choice(list(self.cache.keys()))
            del self.cache[random_key]
            self.cache[idx] = item
        
        return item
    

### End Datasets




### Dataloaders

def create_image_loaders(batch_size=32, image_size=128, shuffle_val=True, data_path=None, 
                         is_midi=False, num_workers=8, val_ratio=0.1, debug=True):
    
    # define transforms
    if is_midi: # midi piano roll images
        train_transforms = midi_transforms(image_size)
        val_transforms = midi_transforms(image_size, random_roll=False)
    else: # for regular images, e.g. from Oxford Flowers dataset
        train_transforms = image_transforms(image_size)
        val_transforms = image_transforms(image_size)
    
    if data_path is None or 'flowers' in data_path.lower(): # fall back to Oxford Flowers dataset
        train_base = datasets.Flowers102(root=data_path, split='train', transform=train_transforms, download=True)
        val_base = datasets.Flowers102(root=data_path, split='val', transform=val_transforms, download=True)
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

### End Dataloaders



# for testing 
if __name__ == "__main__":
    # test the MIDIImageDataset class
    dataset = MIDIImageDataset(debug=True)
    print(f"Number of images in dataset: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image size: {img.size}, Label: {label}")
