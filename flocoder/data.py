import os
import re
from pathlib import Path
from PIL import Image
import random
import torch 
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
from tqdm import tqdm

from .pianoroll import midi_to_pr_img
from .general import ldcfg

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


class MyRGBToGrayscale:
    """Sum RGB channels equally (no luminance weighting)"""
    def __call__(self, tensor):
        # Equal weighting: red and green of same intensity → same grayscale value
        grayscale = tensor.sum(dim=0, keepdim=True).clamp(0, 1.0)
        return grayscale

class BinaryGate:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, tensor):
        return torch.where(tensor >= self.threshold, torch.ones_like(tensor), torch.zeros_like(tensor))


def midi_transforms(image_size=128, random_roll=True, grayscale=False, binary_thresh=0.3):
    """Standard image transformations for training and validation."""
    transform_list = [
        RandomRoll() if random_roll else None,
        transforms.RandomCrop(image_size),
        transforms.ToTensor()]
    if grayscale: transform_list.append(MyRGBToGrayscale())
    if binary_thresh > 0: transform_list.append(BinaryGate(binary_thresh))
    return transforms.Compose([t for t in transform_list if t is not None])


def image_transforms(image_size=128, 
                     #means=[0.485, 0.456, 0.406], # as per common ImageNet metrics
                     #stds=[0.229, 0.224, 0.225]):
                     #means=[0.4524607, 0.39065456, 0.30743122], # what I measured from Oxford Flowers
                     #stds=[0.29211318, 0.24239005, 0.27345273],
                     means = [0.5, 0.5, 0.5],   # this works for TY's flow code
                     stds = [0.5, 0.5, 0.5]):
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
                 redraw_blank=True,  # if (tranformed) image is blank, get a new one
                 redraw_tol= 50, # minimum number of nonzero pixels so that image won't be re-'drawn' from dataset
                 debug=True):

        self.files = file_list
        # Apply split if needed
        if split != 'all' and len(file_list) > 0:
            random.seed(seed)  # For reproducibility
            all_files = file_list.copy()  # Make a copy to avoid modifying the original
            random.shuffle(all_files)
            split_idx = int(len(all_files) * (1 - val_ratio))
            self.files = all_files[:split_idx] if split=='train' else all_files[split_idx:]

        self.actual_len = len(self.files)
        self.images = [None]*self.actual_len
        self.transform = transform
        self.redraw_blank = redraw_blank
        self.max_redraws, self.redraw_tol = 15, redraw_tol

        if debug: print(f"Dataset contains {self.actual_len} images")
        
    def __len__(self): return self.actual_len 
        
    def __getitem__(self, idx):
        actual_idx = idx % self.actual_len
        if self.images[actual_idx] is None:
            self.images[actual_idx] = Image.open(self.files[actual_idx]).convert('RGB')
        img = self.images[actual_idx]
    
        if self.transform: img = self.transform(img)
    
        redraw_attempts = 0
        while (self.redraw_blank and redraw_attempts < self.max_redraws and img.abs().sum() < self.redraw_tol):
            idx = random.randint(0, self.actual_len - 1)
            img, _ = self.__getitem__(idx)
            redraw_attempts += 1

        if redraw_attempts >= self.max_redraws: print("dataset: WARNING: hit max_redraws")
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
                 total_only=False,    # if true, it will only keep the "_TOTAL_" version of each song
                 download=True,      # if true, it will download the datase -- leave this on for now
                 config=None,        # additional config info (starting to get too many kwargs!)
                 redraw_blank=True,  # if (transformed) image is blank, get a new one
                 debug=True):
        self.add_onsets = ldcfg(config,'add_onsets', True)
        self.grayscale = ldcfg(config,'in_channels', 3) == 1
        self.redraw_blank = redraw_blank
        
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
        if total_only:  # don't need this anymore. was used previously to avoid data leakage
            print("MIDIImageDataset: total_only = True. Grabbing only files with _TOTAL in the name")
            self.midi_img_file_list = [f for f in self.midi_img_file_list if '_TOTAL' in f]
        if debug: print(f"len(midi_img_file_list): {len(self.midi_img_file_list)}")

        if split != 'all': # split POP909 by directory names
            import re
            # Extract unique directory numbers
            dir_nums = set()
            for filepath in self.midi_img_file_list:
                match = re.search(r'/(\d{3})/', filepath)
                if match: dir_nums.add(int(match.group(1)))
            dir_nums = sorted(dir_nums)
            if debug: print(f"Found {len(dir_nums)} unique directories: {min(dir_nums)} to {max(dir_nums)}")

            # Split directories by val_ratio
            random.seed(seed)
            split_idx = int(len(dir_nums) * (1 - val_ratio))
            #selected_dirs = sorted(list(set(dir_nums[:split_idx] if split == 'train' else dir_nums[split_idx:])))
            selected_dirs = dir_nums[:split_idx] if split == 'train' else dir_nums[split_idx:] # this is simpler
            if split=='val': print("val: len(selected_dirs) =",len(selected_dirs),", selected_dirs = ",selected_dirs)

            
            # Filter files by selected directories
            self.midi_img_file_list = [f for f in self.midi_img_file_list
                                     if re.search(r'/(\d{3})/', f) and int(re.search(r'/(\d{3})/', f).group(1)) in selected_dirs]

            if debug: print(f"Final file count for split '{split}': {len(self.midi_img_file_list)}")


        super().__init__(self.midi_img_file_list, transform=transform,   # We inherit from ImageListDataset
                         split='all', val_ratio=val_ratio, seed=seed, debug=debug) # split='all' since we already did the split
 
    def convert_one(self, midi_file, debug=True):
        if debug: print(f"Converting {midi_file} to image")
        midi_to_pr_img(midi_file, self.midi_img_dir, show_chords=False, all_chords=None, 
                          chord_names=None, filter_mp=True, add_onsets=self.add_onsets,
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

        # Pass through all attributes from base_dataset
        for attr in dir(base_dataset):
            if not attr.startswith('__') and not callable(getattr(base_dataset, attr)) and not hasattr(self, attr):
                try: setattr(self, attr, getattr(base_dataset, attr))
                except (AttributeError, TypeError): pass

    def __iter__(self):
        while True: yield self.dataset[random.randint(0, self.actual_len - 1)]



class PreEncodedDataset(Dataset):
    """Loads pre-encoded latent tensors with robust class handling"""
    def __init__(self, data_dir, max_cache_items=10000, n_classes=None):
        data_dir = os.path.expanduser(data_dir)
        self.data_dir = Path(data_dir)
        print(f"PreEncodedDataset: loading from {data_dir}")

        # Check for class directories (numeric directories)
        # TODO: note that some of these may not be for classes, they may just be for convenience
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.isdigit()]

        self.files, self._labels = [], []
        self.has_classes = len(class_dirs) > 0 
        if n_classes is not None and n_classes == 0: self.has_classes=False  # directories aren't classes

        if self.has_classes:
            # Class directories found - use class-based structure
            self.n_classes = len(class_dirs)
            self.class_to_idx = {int(d.name): i for i, d in enumerate(sorted(class_dirs))}

            # Collect files from each class directory
            for class_dir in sorted(class_dirs):
                class_idx = self.class_to_idx[int(class_dir.name)]
                _, class_files = fast_scandir(str(class_dir), ['pt'])
                self.files.extend([Path(f) for f in class_files])
                self._labels.extend([class_idx] * len(class_files))
        else:
            # No class structure - get files from subdirs or flat structure
            subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            if subdirs:
                for subdir in subdirs:
                    _, subdir_files = fast_scandir(str(subdir), ['pt'])
                    self.files.extend([Path(f) for f in subdir_files])
            else:
                _, flat_files = fast_scandir(str(self.data_dir), ['pt'])
                self.files = [Path(f) for f in flat_files]
            self.n_classes = 0
            self._labels = [0] * len(self.files)  # All zeros for no-class datasets

        self.actual_len = len(self.files)  # Store for reference
        self.cache = {}  # Initialize memory cache
        self.max_cache_items = max_cache_items

        # Log basic dataset info
        print(f"Found {self.actual_len} samples" +
              (f" across {self.n_classes} classes" if self.has_classes else ""))

    def __len__(self):
        return self.actual_len

    def __getitem__(self, idx):
        if idx in self.cache: return self.cache[idx]  # Return cached item if available

        file_path = self.files[idx]
        class_idx = self._labels[idx]

        try:
            encoded = torch.load(file_path, map_location='cpu')
            item = (encoded, torch.tensor(class_idx, dtype=torch.long))

            # Update cache with simple random replacement policy
            if len(self.cache) < self.max_cache_items:
                self.cache[idx] = item
            elif random.random() < 0.01:  # 1% chance to replace existing item
                random_key = random.choice(list(self.cache.keys()))
                del self.cache[random_key]
                self.cache[idx] = item

            return item
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros with same shape as other items or fallback shape
            fallback = next(iter(self.cache.values()))[0] if self.cache else torch.zeros(4, 16, 16)
            return torch.zeros_like(fallback), torch.tensor(0)




class ColorAwareDataset(Dataset):
    # NOTE: This doesn't help and could/should be deleted
    """Used only for VQGAN & Oxford Flowers: This is a quickie hack I made. 
    Oxford Flowers is low in blue and high in red, 
    so this is an attempt to balance out the data distribution for training.
    What it does: If a given image is high in red and low in blue, it will be rejected (and replaced) with a certain probability.
    """
    def __init__(self, base_dataset, 
                 # all the following numbers were made up, based on a bit of measurement. adjust as needed.
                 red_thresh=0.4, 
                 blue_thresh=0.4, 
                 reject_prob=0.4, 
                 max_attempts=10):
        self.base_dataset = base_dataset
        self.red_thresh = red_thresh
        self.blue_thresh = blue_thresh
        self.reject_prob = reject_prob
        self.max_attempts = max_attempts

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        for _ in range(self.max_attempts):
            img, label = self.base_dataset[idx]
            arr = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            if arr.max() > 1.0: arr = arr / 255.0  # handle PIL images
            r, g, b = arr[0].mean(), arr[1].mean(), arr[2].mean()
            # If high red and low blue, maybe reject
            if r > self.red_thresh and b < self.blue_thresh and np.random.rand() < self.reject_prob:
                idx = np.random.randint(0, len(self.base_dataset))
                continue
            return img, label
        # If we failed to find a good one, just return the last
        return img, label



### End Datasets




### Dataloaders

def create_image_loaders(batch_size=32, image_size=128, shuffle_val=True, data_path=None, 
                         is_midi=False, num_workers=8, val_ratio=0.1, 
                         config=None, # more options to pass
                         debug=True):
    
    # define transforms
    if is_midi: # midi piano roll images
        grayscale = ldcfg(config,'in_channels', 3) == 1
        if debug: print(f"\n--setting grayscale = {grayscale}\n")
        train_transforms = midi_transforms(image_size, grayscale=grayscale)
        val_transforms = midi_transforms(image_size, random_roll=False, grayscale=grayscale)
    else: # for regular images, e.g. from Oxford Flowers dataset
        train_transforms = image_transforms(image_size)
        val_transforms = image_transforms(image_size)
    
    if data_path is None or 'flowers' in data_path.lower(): # fall back to Oxford Flowers dataset
        train_base = ColorAwareDataset(datasets.Flowers102(root=data_path, split='train', transform=train_transforms, download=True))
        val_base = ColorAwareDataset(datasets.Flowers102(root=data_path, split='val', transform=val_transforms, download=True))
    elif 'stl10' in str(data_path).lower():
        train_base = datasets.STL10(root=data_path, split='train', transform=train_transforms, download=True)
        val_base = datasets.STL10(root=data_path, split='test', transform=val_transforms, download=True)
    elif 'food101' in str(data_path).lower():
        train_base = datasets.Food101(root=data_path, split='train', transform=train_transforms, download=True)
        val_base = datasets.Food101(root=data_path, split='test', transform=val_transforms, download=True)
    elif is_midi:
        train_base = MIDIImageDataset(split='train', transform=train_transforms, download=True, val_ratio=val_ratio, config=config)
        val_base = MIDIImageDataset(split='val', transform=val_transforms, download=True, val_ratio=val_ratio, config=config)
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
