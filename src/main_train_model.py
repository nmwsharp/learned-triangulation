import sys, os, datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# add the path to the files from this project
sys.path.append(os.path.dirname(__file__))

import world
import utils
import losses
import mesh_utils
from utils import *
import data_utils
import train_utils
from point_tri_net import PointTriNet_Mesher

### Experiment options

args = world.ArgsObject()
world.args = args

# System parameters
args.experiment_name = "fit_model"
args.run_name = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
experiments_dir = os.path.join(os.path.dirname(__file__), "..")
args.run_dir = os.path.join(os.path.dirname(__file__), "../training_runs", args.run_name)
args.log_dir = os.path.join(args.run_dir, "logs")
args.dataset_dir = os.path.join(experiments_dir, "..", "data")
args.debug_checks = False
args.disable_cuda = False

# set some defaults
set_args_defaults(args)
world.device = args.device
world.dtype = args.dtype

# Experiment parameters
args.load_weights = None

args.w_watertight = 1.0
args.w_dist_surf_tri = 1.0
args.w_dist_tri_surf = 1.0
args.w_overlap_kernel = 0.01

# Algorithm parameters
args.n_mesh_rounds = 5

# Training parameters
args.epochs = 3                         # Number of epochs to train for
args.lr = 1e-4                          # "Learning rate"
args.lr_decay = .5                      # How much to decrease the learning rate by, applied every decay_step samples (lr = lr * lr_decay)
args.decay_step = 10e99                 # Decay lr after processing this many samples (_not_ batches, samples)
args.batch_size = 1                     # Batch size (note accum parameter below)
args.batch_accum = 8                    # Accumulate over this many batches before stepping gradients
args.eval_every = 2048                  # Evaluate on the validation set after this many samples
args.eval_size = 512                    # Use this much of the validation set to evaluate (if less than full validation set size)

print("Beginning run " + str(args.run_name))

# Ensure the run directory exists
ensure_dir_exists(args.run_dir)

## Initialize the tensorboard writer
world.tb_writer = SummaryWriter(log_dir=args.log_dir)

# Load the dataset
T = []
train_dataset = data_utils.PointSurfaceDataset(dir_with_meshes="data/train/", transforms=T)
val_dataset = data_utils.PointSurfaceDataset(dir_with_meshes="data/val/", transforms=T)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=world.args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    # collate_fn=train_dataset.collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=world.args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    # collate_fn=val_dataset.collate_fn
)

# Construct a model
model = PointTriNet_Mesher()

if args.load_weights:
    weights_path = os.path.join(world.args.load_weights,"model_state_dict.pth")
    model.load_state_dict(torch.load(weights_path))

# Calls the model on a batch
def call_model_fn(model, batch, trainer=None):
    points = batch['vert_pos'].to(world.device)
    candidate_triangles, candidate_probs, proposal_triangles, proposal_probs = model.predict_mesh(points, n_rounds=args.n_mesh_rounds, sample_last=True)
    return {
            "candidates" : candidate_triangles, "probs": candidate_probs, 
            "proposals" : proposal_triangles, "proposal_probs" : proposal_probs
           }

# Construct a loss
def loss_fn(batch, model_outputs, viz_extra=False, trainer=None):
    B = batch['vert_pos'].shape[0]

    # Data from the sample
    vert_pos_batch = batch['vert_pos'].to(world.device)
    surf_pos_batch = batch['surf_pos'].to(world.device)
    surf_normal_batch = batch['surf_normal'].to(world.device)

    # Outputs from model
    all_candidates = model_outputs['candidates']
    all_candidate_probs = model_outputs['probs']
    all_proposals = model_outputs['proposals']
    all_proposal_probs = model_outputs['proposal_probs']

    # Accumulate loss
    need_grad = all_candidate_probs.requires_grad
    total_loss = torch.tensor(0.0, dtype=vert_pos_batch.dtype, device=vert_pos_batch.device, requires_grad=need_grad)

    # Evaluate loss one batch entry at a time
    for b in range(B):

        vert_pos = vert_pos_batch[b, :]
        candidates = all_candidates[b,:,:]
        candidate_probs = all_candidate_probs[b, :]
        proposals = all_proposals[b,:,:]
        proposal_probs = all_proposal_probs[b, :]
        
        surf_pos = surf_pos_batch[b, :]
        surf_normal = surf_normal_batch[b, :]

        # Add all the terms
        loss_terms = losses.build_loss(args, vert_pos, candidates, candidate_probs, surf_pos=surf_pos, surf_normal=surf_normal, n_sample=1000)

        loss_terms["proposal_match"] = losses.match_predictions(
            candidates, candidate_probs.detach(),
            proposals, proposal_probs)

        this_loss = torch.tensor(0.0, dtype=vert_pos_batch.dtype, device=vert_pos_batch.device, requires_grad=need_grad)
        for t in loss_terms:
            this_loss = this_loss + loss_terms[t]
            
        
        # Log some stats
        if trainer is not None:
            if trainer.training:
                prefix = "train_" 
                it = trainer.curr_iter + b
            else:
                prefix = "val_" 
                it = trainer.eval_iter + b

            # log less
            if it % 10 == 0:
               
                for t in loss_terms:
                    world.tb_writer.add_scalar(prefix + t, loss_terms[t].item(), it)

                world.tb_writer.add_scalar(prefix + "sample loss", this_loss.item(), it)

                if it % 1000 == 0:
                    world.tb_writer.add_histogram(prefix + 'triangle_probs', candidate_probs.detach(), it)
                    world.tb_writer.add_histogram(prefix + 'triangle_proposal_probs', proposal_probs.detach(), it)

                world.tb_writer.add_scalar(prefix + "prob mean", torch.mean(candidate_probs).item(), it)
                world.tb_writer.add_scalar(prefix + "prob stddev", torch.std(candidate_probs).item(), it)

                if not trainer.training:
                    trainer.add_eval_stat_entry("prob mean", torch.mean(candidate_probs).item())
                    trainer.add_eval_stat_entry("prob std", torch.std(candidate_probs).item())
                    for t in loss_terms:
                        trainer.add_eval_stat_entry(t, loss_terms[t].item())

        total_loss = total_loss + this_loss

    return total_loss / B

# Train
with torch.autograd.set_detect_anomaly(world.debug_checks):

    trainer = train_utils.MyTrainer(
        args=world.args,
        model=model,
        call_model_fn=call_model_fn,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.train()
