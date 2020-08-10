import torch
import numpy as np

import world
import utils
import knn
from utils import *
import mesh_utils

import torch_scatter


# Distance from a known surface to a generated triangulation with probabilities
def dist_surface_to_triangle_probs(gen_verts, gen_faces, gen_face_probs, n_sample_pts=None, mesh=None, surf_samples=None):
    if gen_faces.shape[0] == 0:
        return torch.tensor(0., device=surf_verts.device)
    
    if surf_samples is not None:
        if mesh is not None or n_sample_pts is not None:
            raise ValueError("bad args!")

    if mesh is not None:
        surf_verts, surf_faces = mesh
        if surf_samples is not None:
            raise ValueError("bad args!")

        # Sample points on the known surfacse
        surf_samples = mesh_utils.sample_points_on_surface(surf_verts, surf_faces, n_sample_pts)


    
    # get a characteristic length
    char_len = utils.norm(surf_samples - torch.mean(surf_samples , dim=0, keepdim=True)).mean()

    # Find the distance to all triangles in the generated surface
    tri_dists = mesh_utils.point_triangle_distance(surf_samples, gen_verts, gen_faces)

    # Sort distances
    k_val = min(32, tri_dists.shape[-1])
    tri_dists_sorted, sorted_inds = torch.topk(tri_dists, largest=False, k=k_val, dim=-1)

    # Compute the likelihoods that each triangle is the nearest for that sample
    sorted_probs = gen_face_probs[sorted_inds]

    prob_none_closer = torch.cat(( # shift to the right, put 1 in first col
                    torch.ones_like(sorted_probs)[:,:1],
                    torch.cumprod(1. - sorted_probs, dim=-1)[:, :-1]  
                    ), dim=-1)

    prob_is_closest = prob_none_closer * sorted_probs

    # Append a last distance very far away, so you get high loss values if nothing is close
    last_prob = 1.0 - torch.sum(prob_is_closest, dim=-1)
    last_dist = char_len * torch.ones(tri_dists.shape[0], dtype=tri_dists.dtype, device=tri_dists.device)
    prob_is_closest = torch.cat((prob_is_closest, last_prob.unsqueeze(-1)), dim=-1)
    prob_is_closest = torch.clamp(prob_is_closest, 0., 1.) # for floating point reasons
    tri_dists_sorted = torch.cat((tri_dists_sorted, last_dist.unsqueeze(-1)), dim=-1)

    
    
    # Use these likelihoods to get expected distance
    expected_dist = torch.sum(prob_is_closest * tri_dists_sorted, dim=-1)
        
    result = torch.mean(expected_dist / char_len)
    return result



# Distance from generated triangulation with probabilities to a known surface
def dist_triangle_probs_to_sampled_surface(surf_pos, surf_normals, gen_verts, gen_faces, gen_face_probs, n_sample_pts=5000):
    if gen_faces.shape[0] == 0:
        return torch.tensor(0., device=surf_verts.device)

    # get a characteristic length
    char_len = utils.norm(surf_pos - torch.mean(surf_pos, dim=0, keepdim=True)).mean()

    # Sample points on the generated triangulation
    samples, face_inds, _ = mesh_utils.sample_points_on_surface(
        gen_verts, gen_faces, n_sample_pts, return_inds_and_bary=True)

    # Likelihoods associated with each point
    point_probs = gen_face_probs[face_inds]

    # Measure the distance to the surface
    knn_dist, neigh = knn.find_knn(samples, surf_pos, k=1)
    neigh_pos = surf_pos[neigh.squeeze(1), :]

    if len(surf_normals) == 0 :
        dists = knn_dist
    else:
        neigh_normal = surf_normals[neigh.squeeze(1), :]
        vecs = neigh_pos - samples
        dists = torch.abs(utils.dot(vecs, neigh_normal))
    
    # Expected distance integral
    exp_dist = torch.mean(dists * point_probs)

    return exp_dist / char_len


# Penalize overcomplete triangulations which overlap themselves by evaluating a spatial kernel at sampled surface
def overlap_kernel(gen_verts, gen_faces, gen_face_probs, n_sample_pts=5000):
    if gen_faces.shape[0] == 0:
        return torch.tensor(0., device=gen_verts.device)

    # Sample points on the generated triangulation
    samples, face_inds, _ = mesh_utils.sample_points_on_surface(
        gen_verts, gen_faces, n_sample_pts, face_probs=(gen_face_probs), return_inds_and_bary=True)

    # Evaluate kernel
    sample_tri_kvals = mesh_utils.triangle_kernel(samples, gen_verts, gen_faces, kernel_height=0.5)

    # Incorporate weights and sum
    sample_tri_kvals_weight = sample_tri_kvals * gen_face_probs.unsqueeze(0)

    # Ideally, all samples should all have one entry with value 1 and 0 for all other entries, so
    # we ask that there be no kernel contribution from any other triangles.
    kernel_sums = torch.sum(sample_tri_kvals_weight, dim=-1)
    kernel_max = torch.max(sample_tri_kvals_weight, dim=-1).values
    scores = (kernel_sums - 1.)**2 + (kernel_max - 1.)**2

    # note that this corresponds to a normalization by the expected area of the surface
    return torch.mean(scores)


def expected_watertight(gen_verts, gen_faces, gen_face_probs):
    if gen_faces.shape[0] == 0:
        return torch.tensor(0., device=gen_verts.device)

    V = gen_verts.shape[0]

    # NOTE V^2 for now

    # Build a list of all 3V halfedges, and the probabilities associated with them
    ind_keys = []
    key_probs = []
    for i in range(3):
        
        indA = gen_faces[:,i]
        indB = gen_faces[:,(i+1)%3]

        ind_min = torch.min(indA, indB)
        ind_max = torch.max(indA, indB)

        ind_key = ind_min * V + ind_max

        ind_keys.append(ind_key)
        key_probs.append(gen_face_probs)

    ind_keys_vv = torch.cat(ind_keys, dim=0)
    key_probs = torch.cat(key_probs, dim=0)

    # compute unique dense edge keys
    _, ind_keys = torch.unique(ind_keys_vv, return_inverse=True)

    # warning: this has all kinds of numerical stability pitfalls
    EPS = 1e-3
    pi = (1. - EPS) * key_probs + EPS * .5 # pull slightly towards .5 to mitigate
    qi = 1. - pi

    # compute probability that there are exactly two incident triangles 
    # prod_qi_edge = torch_scatter.scatter_mul(qi, ind_keys)
    prod_qi_edge = torch.exp(torch_scatter.scatter_add(torch.log(qi), ind_keys))
    prob1_this = pi * prod_qi_edge[ind_keys] / qi # probability that this halfedge is the only one incident on tri
    prob1_edge = torch_scatter.scatter_add(prob1_this, ind_keys) # probability that each edge has exactly one tri incident
    prob1_other = (prob1_edge[ind_keys] - prob1_this) / qi # probabilty that there is exactly one tri other than this one
    
    # expected halfedges without unique twin
    loss = torch.sum(pi * (1. - prob1_other)) / (torch.sum(pi) + 1e-4)
    
    return loss



def match_predictions(candA, predA, candB, predB):
    
    candA, predA = mesh_utils.uniqueify_triangle_prob_batch(candA.unsqueeze(0), predA.unsqueeze(0))
    candB, predB = mesh_utils.uniqueify_triangle_prob_batch(candB.unsqueeze(0), predB.unsqueeze(0))
    candA = candA.squeeze(0)
    predA = predA.squeeze(0)
    candB = candB.squeeze(0)
    predB = predB.squeeze(0)

    # form combined list
    cands = torch.cat((candA, candB), dim=0)
    preds = torch.cat((predA, predB), dim=0)

    # unique an compute average
    u_cands, u_inds = torch.unique(cands, dim=0, return_inverse=True)
    u_mean = torch_scatter.scatter_mean(preds, u_inds)
    n_shared = candA.shape[0] + candB.shape[0] - u_cands.shape[0]

    # difference from mean
    diffs = preds - u_mean[u_inds]

    return torch.sum(diffs**2) / (n_shared + 1e-3)


def build_loss(args, vert_pos, candidates, candidate_probs, surf_pos=None, surf_normal=None, gt_tris=None, gt_probs=None, n_sample=None, dist_surf_subsample_factor=10):

    loss_terms = {}

    if hasattr(args, "w_dist_surf_tri") and args.w_dist_surf_tri > 0.:
        loss_terms['dist_surf_tri'] = args.w_dist_surf_tri * \
            dist_surface_to_triangle_probs(vert_pos, candidates, candidate_probs, surf_samples=surf_pos[...,::dist_surf_subsample_factor,:])

    if hasattr(args, "w_dist_tri_surf") and args.w_dist_tri_surf > 0.:
        loss_terms['dist_tri_surf'] = args.w_dist_tri_surf * \
            dist_triangle_probs_to_sampled_surface(surf_pos, surf_normal, vert_pos, candidates, candidate_probs, n_sample_pts=n_sample)

    if hasattr(args, "w_overlap_kernel") and args.w_overlap_kernel> 0.:
        loss_terms['overlap_kernel'] = args.w_overlap_kernel* \
            overlap_kernel(vert_pos, candidates, candidate_probs, n_sample_pts=n_sample)
    
    if hasattr(args, "w_watertight") and args.w_watertight > 0.:
        loss_terms['watertight'] = args.w_watertight * \
            expected_watertight(vert_pos, candidates, candidate_probs)

    return loss_terms
