import sys
import random
import argparse
import numpy as np
import sys
import os
import gc

import utils

from scipy.io import loadmat
from scipy import spatial
import meshio
from plyfile import PlyData


"""
Generate training data in the form of points for meshes in local neighborhoods.
"""

sys.setrecursionlimit(10000)

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def generate_sample_counts(entries, total_count):

    counts = np.zeros(len(entries), dtype=int)
    for i in range(total_count):
        ind = np.random.randint(len(entries))
        counts[ind] += 1

    return counts

def area_normals(verts, faces):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = np.cross(vec_A, vec_B)
    return raw_normal

    
def uniform_sample_surface(verts, faces, n_pts):

    areaN = area_normals(verts, faces)
    face_areas = 0.5 * np.linalg.norm(areaN, axis=-1)

    # chose which faces
    face_inds = np.random.choice(faces.shape[0], size=(n_pts,), replace=True, p=face_areas/np.sum(face_areas))

    # Get barycoords for each sample
    r1_sqrt = np.sqrt(np.random.rand(n_pts))
    r2 = np.random.rand(n_pts)
    bary_vals = np.zeros((n_pts, 3))
    bary_vals[:, 0] = 1. - r1_sqrt
    bary_vals[:, 1] = r1_sqrt * (1. - r2)
    bary_vals[:, 2] = r1_sqrt * r2

    return face_inds, bary_vals

def get_samples(verts, faces, n_pts):

    face_inds, bary_vals = uniform_sample_surface(verts, faces, n_pts)
    # face_normals = igl.per_face_normals(verts, faces, np.array((0., 0., 0.,)))
    areaN = area_normals(verts, faces)
    face_normals = areaN / np.linalg.norm(areaN, axis=-1)[:,np.newaxis]

    positions = np.sum(bary_vals[:,:,np.newaxis] * verts[faces[face_inds, :]], axis=1)
    normals = face_normals[face_inds]
    
    return positions, normals

def main():

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument('--input_dir', type=str, required=True, help='path to the files')
    parser.add_argument('--output_dir', type=str, required=True, help='where to put results')
    parser.add_argument('--n_samples', type=int, required=True, help='number of neighborhoods to sample')

    parser.add_argument('--neigh_size', type=int, default=256, help='number of vertices to sample in each region')
    parser.add_argument('--surface_size', type=int, default=1024, help='number of points to use to represent the surface')
    parser.add_argument('--model_frac', type=float, default=0.25, help='what fraction of the shape each neighborhood should be')
    
    parser.add_argument('--n_add', type=float, default=0.0, help='fraction of noise points to add')
    parser.add_argument('--on_surface_dev', type=float, default=0.02, help='')

    parser.add_argument('--polyscope', action='store_true', help='viz')

    # Parse arguments
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    # Load the list of meshes
    meshes = []
    for f in os.listdir(args.input_dir):
        meshes.append(os.path.join(args.input_dir, f))

    print("Found {} mesh files".format(len(meshes)))
    random.shuffle(meshes)
    counts = generate_sample_counts(meshes, args.n_samples)
    i_sample = 0


    def process_file(i_mesh, f):
        nonlocal i_sample

        # Read the mesh

        # libigl loader seems to leak memory in loop?
        # verts, faces = utils.read_mesh(f)

        plydata = PlyData.read(f)
        verts = np.vstack((
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        )).T
        tri_data = plydata['face'].data['vertex_indices']
        faces = np.vstack(tri_data)


        # Compute total sample counts
        n_vert_sample_tot = int(args.neigh_size / args.model_frac * (1. - args.n_add))
        n_surf_sample_tot = int(args.surface_size / (args.model_frac))

        # sample points
        vert_sample_pos, vert_sample_normal = get_samples(verts, faces, n_vert_sample_tot)

        if(args.n_add > 0):
            n_vert_sample_noise = int(args.neigh_size / args.model_frac * (args.n_add))
            vert_sample_noise_pos, vert_sample_noise_normal = get_samples(verts, faces, n_vert_sample_noise)
            vert_sample_noise_pos += np.random.randn(n_vert_sample_noise, 3) * args.on_surface_dev

            vert_sample_pos = np.concatenate((vert_sample_pos, vert_sample_noise_pos), axis=0)
            vert_sample_normal = np.concatenate((vert_sample_normal, vert_sample_noise_normal), axis=0)


        surf_sample_pos, surf_sample_normal = get_samples(verts, faces, n_surf_sample_tot)
        
        # Build nearest-neighbor structure
        kd_tree_vert = spatial.KDTree(vert_sample_pos)
        kd_tree_surf = spatial.KDTree(surf_sample_pos)

        # Randomly sample vertices
        last_sample = i_sample + counts[i_mesh]
        while i_sample < last_sample:

            print("generating sample {} / {}  on mesh {}".format(i_sample, args.n_samples, f))

            # Random vertex
            ind = np.random.randint(vert_sample_pos.shape[0])
            center = surf_sample_pos[ind, :]

            _, neigh_vert = kd_tree_vert.query(center, k=args.neigh_size)
            _, neigh_surf = kd_tree_surf.query(center, k=args.surface_size)

            result_vert_pos = vert_sample_pos[neigh_vert, :]
            result_vert_normal = vert_sample_normal[neigh_vert, :]
            result_surf_pos = surf_sample_pos[neigh_surf, :]
            result_surf_normal = surf_sample_normal[neigh_surf, :]

            # Write out the result
            out_filename = os.path.join(args.output_dir, "neighborhood_points_{:06d}.npz".format(i_sample))
            np.savez(out_filename, vert_pos=result_vert_pos, vert_normal=result_vert_normal, surf_pos= result_surf_pos, surf_normal=result_surf_normal)

            i_sample = i_sample + 1


    for i_mesh, f in enumerate(meshes):
        process_file(i_mesh, f)



if __name__ == "__main__":
    main()
