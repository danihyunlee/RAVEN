import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*", "*.npz")) \
                            if dataset_type in f]
        self.img_size = img_size
        
        # Fix for Python 3 pickle encoding issues
        try:
            self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), 
                                    allow_pickle=True, encoding='latin1')
        except UnicodeError:
            # Fallback for newer Python versions
            try:
                self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), 
                                        allow_pickle=True, encoding='bytes')
            except Exception as e:
                print(f"Warning: Could not load embedding.npy with encoding='bytes': {e}")
                # Last resort: try without encoding
                self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), 
                                        allow_pickle=True)
        
        # Debug: Print embedding dictionary info
        embedding_dict = self.embeddings.item() if hasattr(self.embeddings, 'item') else self.embeddings
        if isinstance(embedding_dict, dict):
            print(f"Loaded embedding dictionary with {len(embedding_dict)} elements")
            print(f"Sample elements: {list(embedding_dict.keys())[:10]}")
        else:
            print(f"Warning: Embeddings is not a dictionary, type: {type(embedding_dict)}")
        
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        
        # Fix for Python 3 pickle encoding issues with data files
        try:
            data = np.load(data_path, allow_pickle=True, encoding='latin1')
        except UnicodeError:
            try:
                data = np.load(data_path, allow_pickle=True, encoding='bytes')
            except Exception as e:
                print(f"Warning: Could not load {data_path} with encoding='bytes': {e}")
                # Last resort: try without encoding
                data = np.load(data_path, allow_pickle=True)
        
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = data["structure"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = list(range(8))  # Convert to list for Python 3 compatibility
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image = []
        for idx in range(0, 16):
            # Use PIL for resizing instead of deprecated scipy.misc.imresize
            img = Image.fromarray(image[idx,:,:].astype(np.uint8))
            img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            resize_image.append(np.array(img_resized))
        resize_image = np.stack(resize_image)
        # image = resize(image, (16, 128, 128))
        # meta_matrix = data["mata_matrix"]

        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        
        # Get the embedding dictionary with robust handling
        try:
            embedding_dict = self.embeddings.item() if hasattr(self.embeddings, 'item') else self.embeddings
        except Exception as e:
            print(f"Warning: Could not extract embedding dictionary: {e}")
            embedding_dict = {}
        
        # Ensure embedding_dict is actually a dictionary
        if not isinstance(embedding_dict, dict):
            print(f"Warning: Embedding dict is not a dictionary, type: {type(embedding_dict)}")
            embedding_dict = {}
        
        for element in structure:
            if element != '/':
                # Handle missing elements in embedding dictionary
                element_embedding = embedding_dict.get(element)
                if element_embedding is None:
                    print(f"Warning: Element '{element}' not found in embedding dictionary. Using zero vector.")
                    print(f"Available elements: {list(embedding_dict.keys())[:10]}...")
                    # Use a zero vector as fallback
                    element_embedding = np.zeros(300, dtype=np.float32)
                else:
                    # Ensure the embedding is a numpy array
                    if not isinstance(element_embedding, np.ndarray):
                        try:
                            element_embedding = np.array(element_embedding, dtype=np.float32)
                        except Exception as e:
                            print(f"Warning: Could not convert embedding for '{element}' to array: {e}")
                            element_embedding = np.zeros(300, dtype=np.float32)
                    
                    # Ensure correct shape
                    if element_embedding.shape != (300,):
                        print(f"Warning: Embedding for '{element}' has wrong shape {element_embedding.shape}, expected (300,)")
                        if len(element_embedding.shape) > 1:
                            element_embedding = element_embedding.flatten()
                        if element_embedding.size != 300:
                            # Pad or truncate to 300 dimensions
                            if element_embedding.size > 300:
                                element_embedding = element_embedding[:300]
                            else:
                                padded = np.zeros(300, dtype=np.float32)
                                padded[:element_embedding.size] = element_embedding
                                element_embedding = padded
                
                embedding[element_idx, :] = torch.tensor(element_embedding, dtype=torch.float)
                element_idx += 1
        if element_idx == 6:
            indicator[0] = 1.
        # if meta_target.dtype == np.int8:
        #     meta_target = meta_target.astype(np.uint8)
        # if meta_structure.dtype == np.int8:
        #     meta_structure = meta_structure.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            # meta_matrix = self.transform(meta_matrix)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
            meta_structure = self.transform(meta_structure)
            # meta_target = torch.tensor(meta_target, dtype=torch.long)
        return resize_image, target, meta_target, meta_structure, embedding, indicator
        
