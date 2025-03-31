import os
from pathlib import Path
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import datasets

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
    """ for custom datasets that are just folders of images """
    def __init__(self, 
                 file_list,      # list of image file paths, e.g. from fast_scandir
                 transform=None, # can specify transforms manually, i.e. outside of dataloader
                 finite=True,    # if false, it will randomly sample from the dataset indefinitely
                 debug=True):
        self.files = file_list
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