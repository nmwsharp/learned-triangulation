import sys
import argparse
import sys
import os

import numpy as np
import torch

import igl
import plyfile
import polyscope

import utils
import mesh_utils
from utils import *
from point_tri_net import PointTriNet_Mesher

def write_ply_points(filename, points):
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats = 'f8, f8, f8')
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el]).write(filename)

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('model_weights_path', type=str, help='path to the model checkpoint')
    parser.add_argument('input_path', type=str, help='path to the input')

    parser.add_argument('--disable_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--sample_cloud', type=int, help='run on sampled points')

    parser.add_argument('--n_rounds', type=int, default=5, help='number of rounds')
    parser.add_argument('--prob_thresh', type=float, default=.9, help='threshold for final surface')

    parser.add_argument('--output', type=str, help='path to save the resulting high prob mesh to. also disables viz')
    parser.add_argument('--output_trim_unused', action='store_true', help='trim unused vertices when outputting')

    # Parse arguments
    args = parser.parse_args()
    set_args_defaults(args)

    viz = not args.output
    args.polyscope = False

    # Initialize polyscope
    if viz:
        polyscope.init()

    # === Load the input
    
    if args.input_path.endswith(".npz"):
        record = np.load(args.input_path)
        verts = torch.tensor(record['vert_pos'], dtype=args.dtype, device=args.device)
        surf_samples = torch.tensor(record['surf_pos'], dtype=args.dtype, device=args.device)

        samples = verts.clone()
        faces = torch.zeros((0,3), dtype=torch.int64, device=args.device)
        
        polyscope.register_point_cloud("surf samples", toNP(surf_samples))

    if args.input_path.endswith(".xyz"):
        raw_pts = np.loadtxt(args.input_path)
        verts = torch.tensor(raw_pts, dtype=args.dtype, device=args.device)

        samples = verts.clone()
        faces = torch.zeros((0,3), dtype=torch.int64, device=args.device)
        
        polyscope.register_point_cloud("surf samples", toNP(verts))

    else:
        print("reading mesh")
        verts, faces = utils.read_mesh(args.input_path)
        print("  {} verts   {} faces".format(verts.shape[0], faces.shape[0]))
        verts = torch.tensor(verts, dtype=args.dtype, device=args.device)
        faces = torch.tensor(faces, dtype=torch.int64, device=args.device)

        # verts = verts[::10,:]
        
        if args.sample_cloud:
            samples = mesh_utils.sample_points_on_surface(verts, faces, args.sample_cloud)
        else:
            samples = verts.clone()

   
    # For very large inputs, leave the data on the CPU and only use the device for NN evaluation
    if samples.shape[0] > 50000:
        print("Large input: leaving data on CPU")
        samples = samples.cpu()
                
    # === Load the model

    print("loading model weights")
    model = PointTriNet_Mesher()
    model.load_state_dict(torch.load(args.model_weights_path))
    
    model.eval()

    with torch.no_grad():

        # Sample lots of faces from the vertices
        print("predicting")
        candidate_triangles, candidate_probs = model.predict_mesh(samples.unsqueeze(0), n_rounds=args.n_rounds)
        candidate_triangles = candidate_triangles.squeeze(0)
        candidate_probs = candidate_probs.squeeze(0)
        print("done predicting")

        # Visualize
        high_prob = args.prob_thresh
        high_faces = candidate_triangles[candidate_probs > high_prob]
        closed_faces = mesh_utils.fill_holes_greedy(high_faces)

        if viz:
            polyscope.register_point_cloud("input points", toNP(samples))

            spmesh = polyscope.register_surface_mesh("all faces", toNP(samples), toNP(candidate_triangles), enabled=False)
            spmesh.add_scalar_quantity("probs", toNP(candidate_probs), defined_on='faces')

            spmesh = polyscope.register_surface_mesh("high prob mesh " + str(high_prob), toNP(samples), toNP(high_faces))
            spmesh.add_scalar_quantity("probs", toNP(candidate_probs[candidate_probs > high_prob]), defined_on='faces')
            
            spmesh = polyscope.register_surface_mesh("hole-closed mesh " + str(high_prob), toNP(samples), toNP(closed_faces), enabled=False)
           
            polyscope.show()


        # Save output
        if args.output:

            high_prob = args.prob_thresh
            out_verts = toNP(samples)
            out_faces = toNP(high_faces)
            out_faces_closed = toNP(closed_faces)

            if args.output_trim_unused:
                out_verts, out_faces, _, _ = igl.remove_unreferenced(out_verts, out_faces)
           
            igl.write_triangle_mesh(args.output + "_mesh.ply", out_verts, out_faces)
            write_ply_points(args.output + "_samples.ply", toNP(samples))
            
            igl.write_triangle_mesh(args.output + "_pred_mesh.ply", out_verts, out_faces)
            igl.write_triangle_mesh(args.output + "_pred_mesh_closed.ply", out_verts, out_faces_closed)
            write_ply_points(args.output + "_samples.ply", toNP(samples))




if __name__ == "__main__":
    main()
