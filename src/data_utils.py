import sys
import os

import torch
import numpy as np

# import polyscope

import world
import utils
from utils import *
import mesh_utils


class PointSurfaceDataset(torch.utils.data.Dataset):

    def __init__(self, dir_with_meshes=None, transforms=[]):
        super(PointSurfaceDataset, self).__init__()

        # Members
        self.mesh_paths = None
        self.transforms = None

        # Constructor
        if dir_with_meshes is not None:

            # Wrap the string if we just got a single directory
            if isinstance(dir_with_meshes, str):
                dir_with_meshes = [dir_with_meshes]

            # Parse out all of the paths
            self.mesh_paths = []
            for d in dir_with_meshes:
                # Just load from a single directory
                for f in os.listdir(d):
                    _, ext = os.path.splitext(f)
                    fullpath = os.path.join(d, f)
                    self.mesh_paths.append(fullpath)


        # Validate that all of the paths are valid, so we fail fast if there's a mistake
        for p in self.mesh_paths:
            if not os.path.isfile(p):
                raise ValueError("Dataset load error: could not find file " + str(p))

        # Save other options
        self.transforms = transforms

        print("\n== PointSurfaceDataset: loaded dataset with {} surfaces .\n".format(len(self.mesh_paths)))

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the mesh
        # (always loads on CPU)
        fullpath = self.mesh_paths[idx]
        record = np.load(fullpath, allow_pickle=True)

        vert_pos = torch.tensor(record['vert_pos'], dtype=world.dtype, device='cpu')
        surf_pos = torch.tensor(record['surf_pos'], dtype=world.dtype, device='cpu')

        if record['vert_normal'] is None:
            vert_normal = torch.zeros((0,3), dtype=world.dtype, device='cpu')
        else:
            vert_normal = torch.tensor(record['vert_normal'], dtype=world.dtype, device='cpu')

        if record['surf_normal'] is None:
            surf_normal = torch.zeros((0,3), dtype=world.dtype, device='cpu')
        else:
            surf_normal = torch.tensor(record['surf_normal'], dtype=world.dtype, device='cpu')

        # Apply transformations
        for transform in self.transforms:
            vert_pos, _, _ = transform(verts=vert_pos)

        return {'vert_pos': vert_pos, 'vert_normal': vert_normal, 'surf_pos' : surf_pos, 'surf_normal' : surf_normal, 'path': fullpath}

