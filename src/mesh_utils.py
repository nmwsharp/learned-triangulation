import math

import torch
from torch.distributions.categorical import Categorical
# import igl
import numpy as np
# import polyscope

import torch_scatter

import world
import utils
import knn
from utils import *


# Cyclically permute the indices of each triangle such that the smallest index comes first
def roll_faces_to_canonical(faces, in_place=False):

    # Don't modify input
    if not in_place:
        faces = faces.clone()

    min_index = torch.argmin(faces, dim=-1)
    for i in range(3):
        mask = min_index == i
        faces[mask] = faces[mask].roll(-i, dims=-1)

    return faces


# Sort the indices of each triangle
def sort_faces_to_canonical(faces, in_place=False):

    # Don't modify input
    if not in_place:
        faces = faces.clone()

    roll_faces_to_canonical(faces, in_place=True)

    # Swap last two indices so largest index comes last
    max_index = torch.argmax(faces, dim=-1)
    mask = max_index == 1
    faces_opp_orient = torch.index_select(faces, 1, torch.tensor([0, 2, 1], device=world.device))
    faces[mask, :] = faces_opp_orient[mask, :]

    return faces


# Return only unique triangles.
# Note, the indices within each face may have been rearranged in the result
# Also, modifies input.
#
# If oriented = True, assumes that the triagnles have been canonically oriented, so
#   - will not modify the orientation
#   - treats differently oriented triangles as different, aka [1,3,5] != [1,5,3]
# if oriented = False, just treats trangles as sets of three indices, and may modify orientation
def uniqueify_faces(faces, oriented=False):

    if oriented:
        roll_faces_to_canonical(faces, in_place=True)
    else:
        sort_faces_to_canonical(faces, in_place=True)

    # Now, simply take smallest
    return torch.unique(faces, sorted=False, dim=0)


def sample_points_on_surface(verts, faces, n_pts, return_inds_and_bary=False, face_probs=None):

    # Choose faces
    face_areas = utils.face_area(verts, faces)
    if face_probs is None:
        # if no probs, just weight directly by areas to uniformly sample surface
        sample_probs = face_areas
        sample_probs = torch.clamp(sample_probs, 1e-30, float('inf')) # avoid -eps area
        face_distrib = Categorical(sample_probs)
    else:
        # if we have face probs, weight by those so we are more likely to sample more probable faces
        sample_probs = face_areas * face_probs
        sample_probs = torch.clamp(sample_probs, 1e-30, float('inf')) # avoid -eps area
        face_distrib = Categorical(sample_probs)

    face_inds = face_distrib.sample(sample_shape=(n_pts,))

    # Get barycoords for each sample
    r1_sqrt = torch.sqrt(torch.rand(n_pts, device=verts.device))
    r2 = torch.rand(n_pts, device=verts.device)
    bary_vals = torch.zeros((n_pts, 3), device=verts.device)
    bary_vals[:, 0] = 1. - r1_sqrt
    bary_vals[:, 1] = r1_sqrt * (1. - r2)
    bary_vals[:, 2] = r1_sqrt * r2

    # Get position in face for each sample
    coords = utils.face_coords(verts, faces)
    sample_coords = coords[face_inds, :, :]
    sample_pos = torch.sum(bary_vals.unsqueeze(-1) * sample_coords, dim=1)

    if return_inds_and_bary:
        return sample_pos, face_inds, bary_vals
    else:
        return sample_pos

# For each point in pointsA, return distances to each point in pointsB
#   pointsA: (A, 3) coords
#   pointsB: (B, 3) coords
#   return: (A, B) dists
def point_point_distances(pointsA, pointsB):

    # Expand so both are NxMx3 tensor
    pointsA_expand = pointsA.unsqueeze(1)
    pointsA_expand = pointsA_expand.expand(-1, points_target.shape[0], -1)
    pointsB_expand = pointsB.unsqueeze(0)
    pointsB_expand = pointsB_expand.expand(pointsA.shape[0], -1, -1)

    diff_mat = pointsA_expand - pointsB_expand
    dist_mat = utils.norm(diff_mat)

    return dist_mat


# For each point in points, returns the distance^2 to each line segment
#   points: (N, 3) coords
#   linesA: (L, 3) coords
#   linesB: (L, 3) coords
#   return: (N, L) dists
def point_line_segment_distances2(points, linesA, linesB):
    n_p = points.shape[0]
    n_l = linesA.shape[0]

    dir_line = utils.normalize(linesB - linesA).unsqueeze(0).expand(n_p, -1, -1)
    vecA = points.unsqueeze(1).expand(-1, n_l, -1) - linesA.unsqueeze(0).expand(n_p, -1, -1)
    vecB = points.unsqueeze(1).expand(-1, n_l, -1) - linesB.unsqueeze(0).expand(n_p, -1, -1)

    # Distances to first endpoint
    dists2 = utils.norm2(vecA)

    # Distances to second endpoint
    dists2 = torch.min(dists2, utils.norm2(vecB))

    # Points within segment
    in_line = (utils.dot(dir_line, vecA) > 0) & (utils.dot(dir_line, vecB) < 0)

    # Distances to line
    line_dists2 = utils.norm2(utils.project_to_tangent(vecA[in_line], dir_line[in_line]))
    dists2[in_line] = line_dists2

    return dists2


# For each point in points, returns the distance to each face in faces
#   points: (N, 3) coords
#   verts: (V, 3) coords
#   faces: (F, 3) inds
#   return: (N, F)
def point_triangle_distance(points, verts, faces):
    n_p = points.shape[0]
    n_f = faces.shape[0]

    # make sure everything is contiguous
    points = points.contiguous()
    verts = verts.contiguous()
    faces = faces.contiguous()

    points_expand = points.unsqueeze(1).expand(-1, n_f, -1)
    face_normals = utils.face_normals(verts, faces)

    # Accumulate distances
    dists2 = float('inf') * torch.ones((n_p, n_f), dtype=points.dtype, device=points.device)

    # True if point projects inside of face in plane
    inside_face = torch.ones((n_p, n_f), dtype=torch.bool, device=points.device)

    for i in range(3):

        # Distance to each of the three edges
        lineA = verts[faces[:, i]]
        lineB = verts[faces[:, (i+1) % 3]]
        dists2 = torch.min(dists2, point_line_segment_distances2(points, lineA, lineB))

        # Edge perp vec (not normalized)
        e_perp = utils.cross(face_normals, lineB - lineA)
        inside_edge = utils.dot(e_perp.unsqueeze(0).expand(n_p, -1, -1), points_expand - lineA.unsqueeze(0).expand(n_p, -1, -1)) > 0
        inside_face = inside_face & inside_edge
        


    dists = torch.sqrt(dists2)

    # For points inside, distance is just normal distance
    point_in_face = verts[faces[:, 0]].unsqueeze(0).expand(n_p, -1, -1)
    inside_face_dist = torch.abs(utils.dot(
        face_normals.unsqueeze(0).expand(n_p, -1, -1)[inside_face],
        points_expand[inside_face] - point_in_face[inside_face]
    ))

    # dists[inside_face] = inside_face_dist
    inside_face_dist_full = torch.zeros_like(dists)
    inside_face_dist_full[inside_face] = inside_face_dist
    dists = torch.where(inside_face, inside_face_dist_full, dists)

    if False:
        polyscope.remove_all_structures()
        
        samp = polyscope.register_point_cloud("points", toNP(points))
        samp.add_scalar_quantity("dist", toNP(dists[:,0]))
        samp.add_scalar_quantity("inside face", toNP(inside_face[:,0].float()))

        tri = polyscope.register_surface_mesh("tri", toNP(verts), toNP(faces[0,:].unsqueeze(0)))
        tri.add_face_vector_quantity("e perp", toNP(e_perp[0,:].unsqueeze(0)))
        tri.add_face_vector_quantity("e vec", toNP((lineB - lineA)[0,:].unsqueeze(0)))
        tri.add_face_vector_quantity("N", toNP((face_normals)[0,:].unsqueeze(0)))

        polyscope.show()

    return dists

# For each point, returns the triangle zone kernel evaluated over all faces at that point
#   points: (N, 3) coords
#   verts: (V, 3) coords
#   faces: (F, 3) inds
#   kernel_height: height of the kernel, as a fraction of the triangle's longest edge
#   return: (N, F)
def triangle_kernel(points, verts, faces, kernel_height=1.0):
    n_p = points.shape[0]
    n_f = faces.shape[0]

    points_expand = points.unsqueeze(1).expand(-1, n_f, -1)
    face_normals = utils.face_normals(verts, faces)

    # Longest edge in each triangle
    # longest_edge = torch.zeros(n_f, dtype=verts.dtype, device=verts.device)

    # True if point projects inside of face in plane
    min_edge_dist = torch.ones((n_p, n_f), dtype=verts.dtype, device=points.device) * float('inf')

    for i in range(3):

        lineA = verts[faces[:, i]]
        lineB = verts[faces[:, (i+1) % 3]]

        # Update longest edge
        # longest_edge = torch.max(longest_edge, torch.norm(lineA - lineB))

        # Edge perp vec 
        e_perp = utils.normalize(utils.cross(face_normals, lineB - lineA))
        edge_inside_dist = utils.dot(e_perp.unsqueeze(0).expand(n_p, -1, -1), points_expand - lineA.unsqueeze(0).expand(n_p, -1, -1))
        # edge_inside_dist = torch.max(dge_inside_dist, torch.tensor(0., device=verts.device))
        min_edge_dist = torch.min(min_edge_dist, edge_inside_dist)

    # normal distance
    point_in_face = verts[faces[:, 0]].unsqueeze(0).expand(n_p, -1, -1)
    normal_face_dist = torch.abs(utils.dot(
        face_normals.unsqueeze(0).expand(n_p, -1, -1),
        points_expand - point_in_face
    ))

    EPS = 1e-8
    k_val = torch.max(torch.tensor(0., device=verts.device), 1. - (normal_face_dist / (min_edge_dist * kernel_height + EPS)))
    k_val = torch.where(min_edge_dist < 0., torch.zeros_like(k_val), k_val)

    return k_val
           
# Different batches will have different numbers of unique triangles, so will also cull extra triangles from some batches, preferring to cull lowest-prob
# Input:
#  candidate_triangles (B, C, 3)
#  candidate_probs (B, C)
def uniqueify_triangle_prob_batch(candidate_triangles, candidate_probs):
    B = candidate_triangles.shape[0]
    C = candidate_triangles.shape[1]

    # TODO some forum posts have indicated that sort becomes faster on the CPU pretty early, might be worth benchmarking transfer to CPU 

    # NOTE These loop over batch dimension, as many routines (esp. unique) don't broadcast like we need
    
    if world.debug_checks:
        for b in range(B):
            utils.check_faces_for_duplicates(candidate_triangles[b,:,:], check_rows=False)

    # Sort indices within each triangle
    candidate_triangles = torch.sort(candidate_triangles, dim=-1).values

    candidate_triangles_list = []
    candidate_probs_list = []
    for b in range(B):

        # Identify unique triangles
        if candidate_triangles.is_cuda:
            _, inverse_inds = torch.unique(candidate_triangles[b], dim=0, return_inverse=True)
        else:
            # the CPU implementation of torch.unique(inverse_inds=True) has some scaling problems;
            _, inverse_inds = np.unique(toNP(candidate_triangles[b]).astype(np.int32), axis=0, return_inverse=True)
            inverse_inds = torch.tensor(inverse_inds, device=candidate_triangles.device, dtype=torch.long)

        # find the largest prob and first entry for each repeat group 
        max_probs, max_entry = torch_scatter.scatter_max(candidate_probs[b, :], inverse_inds)
        max_entry = max_entry.detach() # needed due to a torch_scatter bug?
        ind_of_max = max_entry[inverse_inds]
        is_max = torch.arange(inverse_inds.shape[0], device=inverse_inds.device) == ind_of_max
        del inverse_inds
        del max_probs
        del max_entry

        # set repeat probs to -1, keeping only the largest prob for repeated triangles
        zeroed_candidate_probs = torch.where(is_max, candidate_probs[b,:], -torch.ones_like(candidate_probs[b,:]))

        # Sort by probabilities, so repeats are now at bottom
        sorted_zeroed_candidate_probs, sort_inds = torch.sort(zeroed_candidate_probs, descending=True)
        candidate_triangles_list.append(candidate_triangles[b, sort_inds, :])
        candidate_probs_list.append(sorted_zeroed_candidate_probs)
        
    # Identify the first index of a repeat in any colum
    repeat_count = max([torch.sum(candidate_probs_list[b] == -1., dim=-1).item() for b in range(B)])

    # Clip out the last repeat_count entries
    u = C-repeat_count
    for b in range(B):
      candidate_triangles_list[b] = candidate_triangles_list[b][:u, :]
      candidate_probs_list[b] = candidate_probs_list[b][:u]
      
    candidate_triangles = torch.stack(candidate_triangles_list)
    candidate_probs = torch.stack(candidate_probs_list)


    if world.debug_checks:
        for b in range(B):
            utils.check_faces_for_duplicates(candidate_triangles[b,:,:], check_rows=True)
    

    return candidate_triangles, candidate_probs

# Different batches will have different numbers of unique triangles, so will also cull extra triangles from some batches
# Input:
#  candidate_triangles (B, C, 3)
def uniqueify_triangle_batch(candidate_triangles):
    B = candidate_triangles.shape[0]
    C = candidate_triangles.shape[1]

    # TODO some forum posts have indicated that sort becomes faster on the CPU pretty early, might be worth benchmarking transfer to CPU 

    # NOTE These loop over batch dimension, as many routines (esp. unique) don't broadcast like we need
    
    if world.debug_checks:
        for b in range(B):
            utils.check_faces_for_duplicates(candidate_triangles[b,:,:], check_rows=False)

    # Sort indices within each triangle
    candidate_triangles = torch.sort(candidate_triangles, dim=-1).values

    candidate_triangles_list = []
    min_length = float('inf')
    for b in range(B):
        unique_triangles = uniqueify_faces(candidate_triangles[b], oriented=False)
        candidate_triangles_list.append(unique_triangles)
        min_length = min((min_length), unique_triangles.shape[0])
    
    # Truncate all in batch to the length of the shortest unique list
    for b in range(B):
        candidate_triangles_list[b] = candidate_triangles[b][:min_length,:]
        
    candidate_triangles = torch.stack(candidate_triangles_list)

    if world.debug_checks:
        for b in range(B):
            utils.check_faces_for_duplicates(candidate_triangles[b,:,:], check_rows=True)

    return candidate_triangles



# Input:
#  candidate_triangles (B, C, 3)
#  candidate_probs (B, C)
def filter_low_prob_triangles(candidate_triangles, candidate_probs, n_keep):
    B = candidate_triangles.shape[0]
    C = candidate_triangles.shape[1]
    u = min(n_keep, C)

    # TODO some forum posts have indicated that sort becomes faster on the CPU pretty early, might be worth benchmarking transfer to CPU 

    # For each group of unique triangles 
    candidate_triangles_list = []
    candidate_probs_list = []
    for b in range(B):

        # Sort by probabilities
        sort_inds = torch.argsort(candidate_probs[b,:], descending=True)
    
        # Clip out all repeated triangles (and possibly some extras from other arrays)
        candidate_triangles_list.append(candidate_triangles[b, sort_inds[:u], :])
        candidate_probs_list.append(candidate_probs[b, sort_inds[:u]])
    
    candidate_triangles = torch.stack(candidate_triangles_list)
    candidate_probs = torch.stack(candidate_probs_list)

    return candidate_triangles, candidate_probs



# Generate V triangles via nearest neighbors
# Note that this will contain lots of duplicates, but that's fine
#  verts (B, V, 3)
def generate_seed_triangles(verts):
    B = verts.shape[0]
    V = verts.shape[1]

    gen_tris = torch.zeros((B,V,3), device=verts.device, dtype=torch.long)

    for b in range(B):
        _, inds = knn.find_knn(verts[b,:], verts[b,:], 2, omit_diagonal=True)
        gen_tris[b, ...] = torch.cat((torch.arange(V, device=verts.device).unsqueeze(-1), inds), dim = -1)

    # gen_probs = torch.rand((B,V), device=verts.device, dtype=verts.dtype)
    gen_probs = 0.5 * torch.ones((B,V), device=verts.device, dtype=verts.dtype)

    return gen_tris, gen_probs


def fill_holes_greedy(in_faces):
    faces = toNP(in_faces).tolist()

    def edge_key(a,b):
        return (min(a,b), max(a,b))
    def face_key(f):
        return tuple(sorted(f))

    edge_count = {}
    neighbors = {}
    all_faces = set()
    def add_edge(a,b):
        if a not in neighbors:
            neighbors[a] = set()
        if b not in neighbors:
            neighbors[b] = set()
        key = edge_key(a,b)
        if key not in edge_count:
            edge_count[key] = 0

        neighbors[a].add(b)
        neighbors[b].add(a)
        edge_count[key] += 1

    def add_face(f):
        for i in range(3):
            a = f[i]
            b = f[(i+1)%3]
            add_edge(a,b)

        all_faces.add(face_key(f)) 

    def face_exists(f):
        return face_key(f) in all_faces

    for f in faces:
        add_face(f)

    # repeated passes (inefficient)
    any_changed = True
    while(any_changed):
        any_changed = False
        new_faces = []

        start_edges = [e for e in edge_count]

        for e in start_edges:
            if edge_count[e] == 1:
                a,b = e
                found = False
                
                # Look single triangle holes
                for s in [a,b]: # one of the verts in this edge
                    if found: break # quit once found
                    o = b if s == a else a # the other vert in this edge
                    for n in neighbors[s]: # a candidate third vertex
                        if found: break # quit once found
                        if n == o: continue # must not be same as edge
                        if face_exists([a,b,n]): continue # face must not exist
                        if (edge_count[edge_key(s,n)] == 1) and (edge_key(o,n) in edge_count) and (edge_count[edge_key(o,n)] == 1):  # must be single hole

                            # accept the new face
                            found = True
                            new_f = [a,b,n]
                            new_faces.append(new_f)
                            add_face(new_f)

        if any_changed:
            # if we found a single hole, look for more
            continue
        
        for e in start_edges:
            if edge_count[e] == 1:
                a,b = e
                found = False

                # Look for matching edge
                for s in [a,b]: # one of the verts in this edge
                    if found: break # quit once found
                    o = b if s == a else a # the other vert in this edge
                    for n in neighbors[s]: # a candidate third vertex
                        if found: break # quit once found
                        if n == o: continue # must not be same as edge
                        if face_exists([a,b,n]): continue # face must not exist
                        if edge_count[edge_key(s,n)] == 1:  # must be boundary edge

                            # accept the new face
                            found = True
                            new_f = [a,b,n]
                            new_faces.append(new_f)
                            add_face(new_f)



        faces.extend(new_faces)

    return torch.tensor(faces, dtype=in_faces.dtype, device=in_faces.device)


        
        
