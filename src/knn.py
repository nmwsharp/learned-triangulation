import torch
#import igl
import numpy as np
import sklearn.neighbors

import world
import utils
from utils import *

import torch_cluster 


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result
    
    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k 
        _, neighbors = kd_tree.query(points_source_np, k=k_search)
        
        if omit_diagonal: 
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")

# For each point in `source`, compute the distance to the nearest point in `target`. Returns the mean of these distances.
def point_cloud_nearest_dist(points_source, points_target):

    # dummy batch IDs
    # source_ids = torch.zeros(points_source.shape[0], dtype=torch.int64, device=points_source.device)
    # target_ids = torch.zeros(points_target.shape[0], dtype=torch.int64, device=points_source.device)

    # get the nearest point in target
    # nearest_ind = torch_cluster.nearest(points_source, points_tarege, source_ids, target_ids)
    nearest_ind = torch_cluster.nearest(points_source, points_target)

    # compute the distances themselves
    nearest_pos = points_target[nearest_ind, :] 
    dists = utils.norm(points_source - nearest_pos)
    return dists


# For face in faces, find the k nearest neighbors in target_points.
# Implemented by first finding the nearest neighbors of the barycenter of the triangle, then discarding the triangles vertices
def face_neighbors(face_centers, faces, target_points, k=10, alternate_centers=None):

    # Get nearby points for each face by nearest neighbor from barycenter, discard neighbors which are the vertices of the triangle
    # (note: the way this is written if the triangle vertices did not end up on the neighbors list, some other neighbors will be
    # discarded so the output tensor has fixed size)
    _, neighbors = find_knn(face_centers, target_points, k+3)  # +3 because we discard below

    # If using alternate centers, includes them here
    if alternate_centers is not None:
        _, alt_neighbors = find_knn(alternate_centers, target_points, k)
        neighbors = torch.cat([neighbors, alt_neighbors], dim=-1)

    # Discard the faces's vertices
    # Build a mask of the indices we want to keep, by masking out neighbors equal to one of the faces vertices
    mask = torch.ones_like(neighbors, dtype=torch.bool)
    for i in range(3):
        ith_vert = faces[:, i]
        ith_mask = (neighbors - ith_vert.unsqueeze(1)) == 0
        # mask = mask & ~ith_mask # old version
        mask = mask.type(torch.uint8) & (~ith_mask).type(torch.uint8)

    # Make sure there are exactly 3 False entries in each mask, by adding Falses to end (which drops farthest points)

    # Note: this uses byte tensors, since where() isn't implemented for bool. But support was added days ago: https://github.com/pytorch/pytorch/pull/26430
    # once that pull makes it in to release, we can just use bool tensors
    mask = mask.to(torch.uint8)

    n_neigh = k + 3
    if alternate_centers is not None:
        n_neigh += k

    n_rounds = 3 if alternate_centers is None else 6
    for i in range(n_rounds):
        false_counts = n_neigh - torch.sum(mask, dim=-1)

        # Alternate version with element from back set to False
        false_back = mask.clone()
        false_back[:, n_neigh-i-1] = False

        replace_row = (false_counts != 3).unsqueeze(-1)
        mask = torch.where(replace_row, false_back, mask)

    # see note above
    mask = mask.to(torch.bool)

    # Must be 0 or reshape below will fail. Luckily, 3 iterations of the loop above will always be enough to make this 0
    # false_counts = n_neigh - torch.sum(mask, dim=-1)
    # print("n bad = " + str(torch.sum(false_counts != 3)))
    # print_info(neighbors[mask], "neigh mask")

    # Take only the masked elements
    result_neighbors = neighbors[mask].reshape((faces.shape[0], -1))

    return result_neighbors


# Generate all neighbors for a triangle, excluding its three vertices
def all_face_neighbors(faces, n_target):
    n_face = faces.shape[0]

    if world.debug_checks:
        utils.check_faces_for_duplicates(faces)

    neighbors = torch.arange(n_target, device=world.device).unsqueeze(0).expand((n_face, -1))

    # Remove the three triangle vertices
    # Build a mask of the indices we want to keep, by masking out neighbors equal to one of the faces vertices
    mask = torch.ones_like(neighbors, dtype=torch.bool)
    for i in range(3):
        ith_vert = faces[:, i]
        ith_mask = (neighbors - ith_vert.unsqueeze(1)) == 0
        # mask = mask & ~ith_mask # old version
        mask = mask.type(torch.uint8) & (~ith_mask).type(torch.uint8)

    # see note above
    mask = mask.to(torch.bool)

    # Take only the masked elements
    result_neighbors = neighbors[mask].reshape((faces.shape[0], -1))

    return result_neighbors


# Inputs: 
#   interp_points: (N,D) locations at which to same values
#   source_points: (M,D) locations at which values are defined
#   source_values: (M,V) value at each location in source_points
#   weight_fn: strategy to weight interpolant
#
# Outputs:
#   (N,V) interpolated values at interp_points 
def interpolate_nearby(interp_points, source_points, source_values, weight_fn='inv_dist', eps=1e-6):
    N = interp_points.shape[0]
    M = source_points.shape[0]

    # TODO could do this with knn for better scaling

    # Expand so both point sets are NxMxD tensor, and value set is NxMxV
    interp_points_expand = interp_points.unsqueeze(1)
    interp_points_expand = interp_points_expand.expand(-1, M, -1)

    source_points_expand = source_points.unsqueeze(0)
    source_points_expand = source_points_expand.expand(N, -1, -1)

    source_values_expand = source_values.unsqueeze(0)
    source_values_expand = source_values_expand.expand(N, -1, -1)

    # Evaluate weight function
    # after, `weights' will be a (NxM) tensor of weights
    if weight_fn == 'inv_dist':

        diff_mat = interp_points_expand - source_points_expand
        dist_mat = norm(diff_mat, highdim=True)
        weights = 1.0 / (torch.pow(dist_mat, 3) + eps)

    else:
        raise ValueError("unrecognized weight function: {}".format(weight_fn))

    # Divide by weight sum to get interpolation coefficients
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Interpolate
    interp_vals = torch.sum(source_values_expand * weights.unsqueeze(-1), dim=1)
    
    return interp_vals


