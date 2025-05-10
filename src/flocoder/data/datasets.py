import os
from pathlib import Path
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import datasets
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
from tqdm import tqdm


from flocoder.data.pianoroll import midi_to_pr_img

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
                 finite=True,    # if false, it will randomly sample from the dataset indefinitely
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
        self.finite = finite
        if debug:
            print(f"Dataset contains {self.actual_len} images")
        
    def __len__(self):
        return self.actual_len if self.finite else 9_999_999_999  # big number
        
    def __getitem__(self, idx):
        actual_idx = idx % self.actual_len  # Use modulo to wrap around the index
        if self.images[actual_idx] is None: # lazy "pre-loading": it will eventualy store all images in CPU memory
            self.images[actual_idx] = Image.open(self.files[actual_idx]).convert('RGB')
        img = self.images[actual_idx]

        if self.transform:
            img = self.transform(img)
        return img, 0  # Return 0 as class label since intended image set (MIDI piano rolls) doesn't have classes
    

class MIDIImageDataset(ImageListDataset):
    """ This renders a midi dataset (POP909 by default) as images """
    def __init__(self, 
                 root=Path.home() / "datasets",  # root directory for the MIDI part of the dataset
                 url = "https://github.com/music-x-lab/POP909-Dataset/raw/refs/heads/master/POP909.zip", # url for downloading the dataset
                 transform=None, # can specify transforms manually, i.e. outside of dataloader
                 split='all',       # 'train', 'val', or 'all'
                 val_ratio=0.1,  # percentage for validation
                 seed=42,        # for reproducibility 
                 finite=True,    # if false, it will randomly sample from the dataset indefinitely
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
                         split=split, val_ratio=val_ratio, seed=seed, finite=finite, debug=debug)
        return 
    

        file_list = []
        # TODO: More code goes here. 

        # Now instantiate parent class

        # next few lines are redundant since parent class already does this
        # self.file_list = file_list
        # self.actual_len = len(self.file_list)
        # self.images = [None]*self.actual_len   
        # 
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


# for testing 
if __name__ == "__main__":
    # test the MIDIImageDataset class
    dataset = MIDIImageDataset(debug=True)
    print(f"Number of images in dataset: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image size: {img.size}, Label: {label}")