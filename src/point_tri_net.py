import sys
import os

import torch
import torch.nn as nn
import numpy as np

import mesh_utils
import utils
import knn

from mini_mlp import MiniMLP


# Encode points with respect to a triangle. For each input point generates (x,y,z,u,v,w) with respect to triangle.
# Inputs:
#  - points_pos (B, Q, K, 3)  positions
#  - query_triangles_pos (B, Q, 3, 3) corner positions
# Outputs:
#  (B, Q, N, 6)
def generate_coords(points_pos, query_triangles_pos):

    EPS = 1e-6
    
    # First, compute and remove the normal component
    area_normals = 0.5 * torch.cross(
        query_triangles_pos[:, :, 1, :] - query_triangles_pos[:, :, 0, :],
        query_triangles_pos[:, :, 2, :] - query_triangles_pos[:, :, 0, :], dim=-1)

    areas = utils.norm(area_normals) + EPS # (B, Q)
    normals = area_normals / areas.unsqueeze(-1) # (B, Q, 3)
    barycenters = torch.mean(query_triangles_pos, dim=2) # (B, Q, 3)
    centered_neighborhood = points_pos - barycenters.unsqueeze(2)
    normal_comp = utils.dot(normals.unsqueeze(2), centered_neighborhood)
    neighborhood_planar = points_pos - normals.unsqueeze(2) * normal_comp.unsqueeze(-1)
    
    # Compute barycentric coordinates in plane
    def coords_i(i):
        point_area = 0.5 * utils.dot(
            normals.unsqueeze(2),
            torch.cross(
              query_triangles_pos[:, :, (i+1) % 3, :].unsqueeze(2) - neighborhood_planar,
              query_triangles_pos[:, :, (i+2) % 3, :].unsqueeze(2) - neighborhood_planar,
                        dim=-1)
        )

        area_frac = (point_area + EPS / 3.) / areas.unsqueeze(-1)
        return area_frac

    BARY_MAX = 5.
    u = torch.clamp(coords_i(0), -BARY_MAX, BARY_MAX)
    v = torch.clamp(coords_i(1), -BARY_MAX, BARY_MAX)
    w = torch.clamp(coords_i(2), -BARY_MAX, BARY_MAX)

    # Compute cartesian coordinates with the x-axis along the i --> j edge
    basisX = utils.normalize(query_triangles_pos[:, :, 1, :] - query_triangles_pos[:, :, 0, :])
    basisY = utils.normalize(torch.cross(normals, basisX))
    x_comp = utils.dot(basisX.unsqueeze(2), centered_neighborhood)
    y_comp = utils.dot(basisY.unsqueeze(2), centered_neighborhood)

    coords = torch.stack((x_comp, y_comp, normal_comp, u, v, w), dim=-1)

    return coords


# Inputs:
# - query_triangles_pos (B, Q, 3, 3)
# - nearby_points_pos (B, Q, K, 3)
# - nearby_triangle_pos (B, Q, K_T, 3, 3)
# - nearby_triangle_probs (B, Q, K_T)
def encode_points_and_triangles(query_triangles_pos, nearby_points_pos, 
    nearby_triangles_pos=None, nearby_triangle_probs=None):

    B = query_triangles_pos.shape[0]
    Q = query_triangles_pos.shape[1]
    K = nearby_points_pos.shape[2]

    have_triangles = (nearby_triangles_pos is not None)
    if have_triangles:
      K_T = nearby_triangles_pos.shape[2]

    # Normalize neighborhood (translation won't matter, but unit scale is nice)
    # note that we normalize vs. the triangle, not vs. the points
    neigh_centers = torch.mean(query_triangles_pos, dim=2) # (B, Q, 3)
    neigh_scales = torch.mean(utils.norm(query_triangles_pos - neigh_centers.unsqueeze(2)), dim=-1) + 1e-5 # (B, Q)
    nearby_points_pos = nearby_points_pos.clone() / neigh_scales.unsqueeze(-1).unsqueeze(-1)
    query_triangles_pos = query_triangles_pos.clone() / neigh_scales.unsqueeze(-1).unsqueeze(-1)
    if have_triangles:
      nearby_triangles_pos = nearby_triangles_pos.clone() / neigh_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Encode the nearby points
    point_coords = generate_coords(nearby_points_pos, query_triangles_pos)

    # Encode the nearby triangles
    if have_triangles:
      tri_coords = generate_coords(nearby_triangles_pos.view(B, Q, K_T*3, 3), query_triangles_pos).view(B, Q, K_T, 3, 6)
      max_vals = torch.max(tri_coords, dim=3).values  # (B, Q, K_T, 6)
      min_vals = torch.min(tri_coords, dim=3).values  # (B, Q, K_T, 6)
      triangle_coords = torch.cat((min_vals, max_vals, nearby_triangle_probs.unsqueeze(-1)), dim=-1)

    if have_triangles:
      return point_coords, triangle_coords
    else:
      return point_coords

# Given probabilities for points 
# Input: 
#   - `query_triangles` (B, Q, 3) indices for the three vertices of triangle
#   - `neighborhoods` (B, Q, K) indices of the neighbors for each query triangle
#   - `point_probs` (B, Q, K) in [0,1] probs for each point connecting to the side
# Output: 
# - (B, Q, O, 3) triangle inds in to points list
# - (B, Q, O) probs for each triangle
def sample_neighbor_tris_from_point(query_triangles, neighborhoods, point_probs, n_output_per_side, random_rate=0.):
    B = point_probs.shape[0]
    Q = point_probs.shape[1]
    K = point_probs.shape[2]
    O = n_output_per_side

    # Zero probs for points which appear in the triangles
    zeros = torch.zeros_like(point_probs)
    sample_probs = point_probs + .0001 # avoid zeros in multinomial
    sample_probs = random_rate * torch.mean(sample_probs, dim=-1, keepdim=True) + (1. - random_rate) * sample_probs # support random_rate option
    for i in range(3):
        remove_pts = query_triangles[:, :, i]
        mask = (neighborhoods == remove_pts.unsqueeze(-1))
        sample_probs = torch.where(mask, zeros, sample_probs)

    # indexed local to this neighborhood
    new_neigh_inds = torch.zeros((B, Q, O), dtype=neighborhoods.dtype, device=neighborhoods.device)

    # loop, multinomial doesn't broadcast (could use view?)
    for b in range(B):
      new_neigh_inds[b,:,:] = torch.multinomial(sample_probs[b,:,:], num_samples=O, replacement=False)
  
    # Global-index the sampled neighbors and gather data
    new_inds = torch.gather(neighborhoods, 2, new_neigh_inds) # (B, Q, O) global indexed
        
    tri_verts = torch.stack((
        query_triangles[:, :, 0].unsqueeze(2).expand(-1, -1, O), 
        query_triangles[:, :, 1].unsqueeze(2).expand(-1, -1, O), 
        new_inds
        ), dim=3)
    
    # Likelihood for the points that were sampled (use actual original likelihood, rather than sample likelihood)
    new_probs = torch.gather(point_probs[:,:,:], 2, new_neigh_inds)
       

    # pull sliiiightly away from 0/1 for numerical stability downstream
    EPS = 1e-4
    new_probs = (1. - EPS) * new_probs + EPS * 0.5

    return tri_verts, new_probs



class PointTriNet(torch.nn.Module):
    def __init__(self, input_dim=3):
        super(PointTriNet, self).__init__()

        use_batch_norm = False
        use_layer_norm = False
        activation = nn.ReLU

        # == The MLPs for the pointnet
        
        self.input_dim = input_dim

        # classification net
        self.neigh_point_feat_class_net = MiniMLP([6 + (self.input_dim-3), 64, 128, 1024], activation=activation, batch_norm=False, layer_norm=False)
        g_dim = 1024
        self.neigh_tri_feat_class_net = MiniMLP([13, 64, 128, 1024], activation=activation, batch_norm=False, layer_norm=False)
        g_dim += 1024

        self.global_feat_class_net = MiniMLP([g_dim, 512, 256, 1], activation=activation, batch_norm=use_batch_norm, layer_norm=use_layer_norm, skip_last_norm=True, dropout=True)

        # suggestion net
        self.neigh_point_feat_sugg_net = MiniMLP([6 + (self.input_dim-3), 64, 64, 128], activation=activation, batch_norm=False, layer_norm=False)
        self.point_sugg_net = MiniMLP([6 + (self.input_dim-3) + 128, 128, 64, 64, 1], activation=activation, batch_norm=use_batch_norm, layer_norm=use_layer_norm, skip_last_norm=True)

    # Input:
    #   - `verts` (B, V, 3) ALL vertices for each shape
    #   - `all_triangle_pos` (B, F, 3, 3) positions for the three vertices of ALL triangles
    #   - `all_triangle_prob` (B, F) current probs for ALL triangles
    #   - `query_triangle_ind` (B, Q, 3) indices for the three vertices of triangle, ordered arbitrarily
    #   - `query_triangle_prob` (B, Q) current probs for the triangles used as queries
    #   - `point_neighbor_ind` (B, Q, K) indices of the neighboring points for each query triangle
    #   - `face_neighbor_ind` (B, Q, K_T) indices of the neighboring triangles for each query triangle
    #
    # Output:
    def forward(self, verts, all_triangle_pos, all_triangle_prob, query_triangle_pos, query_triangle_ind, query_triangle_prob, point_neighbor_ind, face_neighbor_ind, preds_per_side):

        if not hasattr(self, 'input_dim'):
            self.input_dim = 3

        D = query_triangle_ind.device
        DT = verts.dtype
        B = point_neighbor_ind.shape[0]
        Q = point_neighbor_ind.shape[1]
        K = point_neighbor_ind.shape[2]
        K_T = face_neighbor_ind.shape[2]

        ## Gather data about neighborhood
        
        point_neighbor_pos = torch.gather(
            verts.unsqueeze(-2).expand(-1, -1, K, -1), 1, 
            point_neighbor_ind.unsqueeze(-1).expand(-1, -1, -1, self.input_dim)
            ) # (B, Q, K, 3)

        face_neighbor_probs = torch.gather(
            all_triangle_prob.unsqueeze(-1).expand(-1, -1, K_T), 1, 
            face_neighbor_ind
            ) # (B, Q, K_T)
        
        face_neighbor_pos = torch.gather(
            all_triangle_pos.unsqueeze(2).expand(-1, -1, K_T, -1, -1), 1,
            face_neighbor_ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3, 3)
            ) # (B, Q, K_T, 3, 3)
       
        # ==========================
        # === Classification
        # ==========================

        # Generate coordinates 
        point_neighbor_coords, face_neighbor_coords = \
            encode_points_and_triangles(query_triangle_pos, point_neighbor_pos[...,:3], face_neighbor_pos, face_neighbor_probs)

        point_neighbor_coords = torch.cat((point_neighbor_coords, point_neighbor_pos[...,3:]), dim=-1) # restore optional latent data


        # Evaluate the pointnet for the point neighbors of each query
        point_features = self.neigh_point_feat_class_net(point_neighbor_coords) # (B, Q, K, feat)
        point_features_max = torch.max(point_features, dim=2).values # (B, Q, feat)

        # Evaluate the pointnet for the face neighbors of each query
        tri_features = self.neigh_tri_feat_class_net(face_neighbor_coords) # (B, Q, K_T, feat)
        tri_features_max = torch.max(tri_features, dim=2).values # (B, Q, feat)
    
        # Combine and take the max
        max_features = torch.cat((point_features_max, tri_features_max), dim=-1) #(B, Q, 2*feat)
          
        # get global features for each output (from both heads)
        global_features_class = self.global_feat_class_net(max_features) # (B, Q, 1)

        # probabilities from classification head
        output_probs = torch.sigmoid(global_features_class.squeeze(-1)) 

        # pull sliiiightly away from 0/1 for numerical stability downstream
        EPS = 1e-4
        output_probs = (1. - EPS) * output_probs + EPS * 0.5

        # happens very rarely due to geometric degeneracies
        output_probs = torch.where(torch.isnan(output_probs), 
                torch.mean(output_probs[~torch.isnan(output_probs)]), output_probs)
        
        # ==========================
        # === Classification
        # ==========================

        # repeat for each of the 3 orientaitons
        gen_tris_list = []
        gen_probs_list = []
        for i in range(3):

            # permute the encoded values, if needed
            # (for the first iteraiton we can just reuse the values from classification prediction)
            if i != 0:
                query_triangle_pos = query_triangle_pos.roll(1, dims=2)
                query_triangle_ind = query_triangle_ind.roll(1, dims=2)
        
                # Generate coordinates 
                point_neighbor_coords = encode_points_and_triangles(query_triangle_pos, point_neighbor_pos[...,:3])
                point_neighbor_coords = torch.cat((point_neighbor_coords, point_neighbor_pos[...,3:]), dim=-1) # restore optional latent data
    

            # Evaluate the pointnet for the point neighbors of each query
            point_features = self.neigh_point_feat_sugg_net(point_neighbor_coords) # (B, Q, K, feat)
            point_features_max = torch.max(point_features, dim=2).values # (B, Q, feat)

            # use these global features to make point predictions
            point_select_inputs = torch.cat((
              point_neighbor_coords,
              point_features_max.unsqueeze(2).expand(-1, -1, K, -1)
            ), dim=-1) # (B, Q, K, 6 + g_feat)

            # generate selection scores at each point
            point_probs = torch.sigmoid(self.point_sugg_net(point_select_inputs).squeeze(-1)) # (B, Q, K)
           

            # Sample new triangles from the vertex likelihoods
            random_rate = 0.25 if self.training else 0. # during training, chose random with small prob to get more divesity
            gen_tris, gen_probs = sample_neighbor_tris_from_point(
              query_triangle_ind, point_neighbor_ind, point_probs, preds_per_side, random_rate)
                

            # modulate point probs by the triangle which generated them
            gen_probs *= output_probs.unsqueeze(-1)
    
            gen_tris_list.append(gen_tris)
            gen_probs_list.append(gen_probs)

        gen_tris = torch.cat(gen_tris_list, dim=2)
        gen_probs = torch.cat(gen_probs_list, dim=2)

        # Collapse all of the new candidates from all of the query triangles
        gen_tris = gen_tris.view(B, -1, 3)
        gen_probs = gen_probs.view(B, -1)

        return output_probs, gen_tris, gen_probs


    # Input:
    #  - query_triangle_ind: (B, Q, 3) indices
    #  - verts: (B, V, 3) positions
    #  - query_triangle_ind: (B, Q) in [0,1]

    #   - as an optimization, neighbors can be passed in if we already have them
    #
    # Output:
    #   - (T) \in [0,1] new likelihood for each face
    def apply_to_candidates(self, query_triangle_ind, verts, query_probs, new_verts_per_edge, k_neigh=64, neighbors_method='generate', return_list=False, split_size=1024*4):
        B = query_triangle_ind.shape[0]
        Q = query_triangle_ind.shape[1]
        V_D = verts.shape[-1]
        D = verts[0].device
        K = k_neigh
        K_T = min(k_neigh, Q-1)

        query_triangles_pos = torch.gather(
            verts[...,:3].unsqueeze(-2).expand(-1, -1, 3, -1), 1, 
            query_triangle_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
            ) # (B, Q, 3, 3)

        barycenters = torch.mean(query_triangles_pos, dim=2)

        # (during training, this should hopefully leave a single chunk, so we get batch statistics
        query_triangle_ind_chunks = torch.split(query_triangle_ind, split_size, dim=1)
        query_triangle_pos_chunks = torch.split(query_triangles_pos, split_size, dim=1)
        query_triangle_prob_chunks = torch.split(query_probs, split_size, dim=1)
        
        # Apply the model
        pred_chunks = []
        gen_tri_chunks = []
        gen_pred_chunks= []
        for i_chunk in range(len(query_triangle_ind_chunks)):
            if(len(query_triangle_ind_chunks) > 1):
                print("chunk {}/{}".format(i_chunk, len(query_triangle_ind_chunks)))

            query_triangle_ind_chunk = query_triangle_ind_chunks[i_chunk]
            query_triangle_pos_chunk = query_triangle_pos_chunks[i_chunk]
            query_triangle_prob_chunk = query_triangle_prob_chunks[i_chunk]

            Q_C = query_triangle_ind_chunk.shape[1]
            barycenters_chunk = torch.mean(query_triangle_pos_chunk, dim=2)

            # Gather neighborhoods of each candidate face
            method = 'brute' if (verts[0].is_cuda and Q < 4096) else 'cpu_kd'

            # Build out neighbors
            point_neighbor_inds = torch.zeros((B, Q_C, K), device=D, dtype=query_triangle_ind.dtype)
            face_neighbor_inds = torch.zeros((B, Q_C, K_T), device=D, dtype=query_triangle_ind.dtype)

            for b in range(B):

                _, point_neighbor_inds_this = knn.find_knn(barycenters_chunk[b,...], verts[b,...,:3], k=K, method=method)
                point_neighbor_inds[b,...] = point_neighbor_inds_this

                _, face_neighbor_inds_this = knn.find_knn(barycenters_chunk[b,...], barycenters[b,...], k=K_T+1, method=method, omit_diagonal=False)
                face_neighbor_inds_this = face_neighbor_inds_this[...,1:] # remove self overlap
                face_neighbor_inds[b,...] = face_neighbor_inds_this


            # Invoke the model
            output_preds_chunk, gen_tri_chunk, gen_pred_chunk = \
                self(verts, query_triangles_pos, query_probs, query_triangle_pos_chunk,
                    query_triangle_ind_chunk, query_triangle_prob_chunk, 
                    point_neighbor_inds, face_neighbor_inds, new_verts_per_edge)

            pred_chunks.append(output_preds_chunk)
            gen_tri_chunks.append(gen_tri_chunk)
            gen_pred_chunks.append(gen_pred_chunk)

        preds = torch.cat(pred_chunks, dim=1)
        gen_tris = torch.cat(gen_tri_chunks, dim=1)
        gen_preds = torch.cat(gen_pred_chunks, dim=1)

        return preds, gen_tris, gen_preds




# Iteratively applies the PointTriNet to generate predictions
class PointTriNet_Mesher(torch.nn.Module):
    def __init__(self, input_dim=3):
        super(PointTriNet_Mesher, self).__init__()

        self.net = PointTriNet(input_dim=input_dim)


    def predict_mesh(self, verts, verts_latent = None, n_rounds=5, keep_faces_per_vert=12, new_verts_per_edge=4, sample_last=False, return_all=False):

        B = verts.shape[0]
        n_keep = keep_faces_per_vert * verts.shape[1]

        # Seed with some random initial triangles
        candidate_triangles, candidate_probs = mesh_utils.generate_seed_triangles(verts[...,:3])
        candidate_triangles, candidate_probs = mesh_utils.uniqueify_triangle_prob_batch(candidate_triangles, candidate_probs)

        if return_all:
            working_tris = []
            working_probs = []
            proposal_tris = []
            proposal_probs = []
       
        for iter in range(n_rounds):
            is_last = iter == n_rounds-1

            # only take gradients on last iter
            with (utils.fake_context() if is_last else torch.autograd.no_grad()):

                # Classify triangles & generate new ones
                new_candidate_probs, gen_tris, gen_probs = self.net.apply_to_candidates(candidate_triangles, verts, candidate_probs, new_verts_per_edge)
                candidate_probs = new_candidate_probs

                if return_all:
                    working_tris.append(candidate_triangles)
                    working_probs.append(candidate_probs)
                    proposal_tris.append(gen_tris)
                    proposal_probs.append(gen_probs)
                        
                if (not is_last):

                    # Union new candidates
                    candidate_triangles = torch.cat((candidate_triangles, gen_tris), dim=1)
                    candidate_probs = torch.cat((candidate_probs, gen_probs), dim=1)
                
                    # Prune out repeats
                    candidate_triangles, candidate_probs = mesh_utils.uniqueify_triangle_prob_batch(candidate_triangles, candidate_probs)
                
                    # Cull low-probability triangles
                    candidate_triangles, candidate_probs = mesh_utils.filter_low_prob_triangles(
                      candidate_triangles, candidate_probs, n_keep)

        if return_all:
            return working_tris, working_probs, proposal_tris, proposal_probs

        if sample_last:
            # Prune out repeats amongst last samples
            gen_tris, gen_probs = mesh_utils.uniqueify_triangle_prob_batch(gen_tris, gen_probs)

            return candidate_triangles, candidate_probs, gen_tris, gen_probs
        else:
            return candidate_triangles, candidate_probs
