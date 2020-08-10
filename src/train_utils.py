import sys
import os
import gc

import numpy as np
import torch

import world
import utils
import data_utils
from utils import toNP

class MyTrainer(object):

    def __init__(
        self, args, model, call_model_fn, loss_fn, train_loader, val_loader, collate_fn=None
    ):

        # Copy parameters
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.call_model_fn = call_model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.collate_fn = collate_fn

        # Some extra training state
        self.best_loss = float("inf")
        self.curr_epoch = 0
        self.curr_iter = 0
        self.training = True # false == eval
        self.eval_iter = 0 # just used as a number for tensorboard logs

        # Stats
        self.running_train_loss = 0.
        self.running_train_loss_count = 0
        self.eval_stats = {} # a dictionary holding lists of stats to track for the eval loop

    # === Utilities
    def run_dir(self):
        return os.path.join(self.args.run_dir)

    def save_dir(self):
        return os.path.join(self.run_dir(), "saved")

    # === Save subroutine
    def save_training_state(self, opt=None, suffix=""):

        # Paths
        this_save_dir = os.path.join(self.save_dir(), suffix)
        print("    --> saving model to {}".format(this_save_dir))
        utils.ensure_dir_exists(this_save_dir)

        # Serialize all the things
        torch.save(self.model, os.path.join(this_save_dir, "model.pth"))
        torch.save(
            self.model.state_dict(), os.path.join(this_save_dir, "model_state_dict.pth")
        )
        if opt is not None:
            torch.save(
                opt.state_dict(), os.path.join(this_save_dir, "opt_state_dict.pth")
            )
        torch.save(self.args, os.path.join(this_save_dir, "args.pth"))
        torch.save(
            {
                'curr_iter': self.curr_iter,
                'curr_epoch': self.curr_epoch,
                'best_loss': self.best_loss,
            },
            os.path.join(this_save_dir, "train_state.pth"))

        with open(os.path.join(this_save_dir, "args.txt"), "w") as text_file:
            text_file.write(world.args_to_str(world.args))

    def train(self):

        # Make sure the model is where it belongs
        if world.device == torch.device("cpu"):
            self.model.cpu()
        else:
            self.model.cuda(world.device)

        # ===  Basic default optimization parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Learning rate schedule
        def lr_lbmd(it):
            lr_clip = 1e-5
            return max(
                self.args.lr_decay ** (int(it / self.args.decay_step)),
                lr_clip / self.args.lr,
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lbmd, last_epoch=-1
        )

        # === Epoch loop
        while self.curr_epoch < self.args.epochs:
            total_train_loss = 0.
            total_train_loss_count = 0

            print("\n\n=== Epoch {} / {}".format(self.curr_epoch, self.args.epochs))

            ib_count = 0
            for batch in self.train_loader:

                self.model.train()
                self.training = True

                # Zero gradients 
                if ib_count >= self.args.batch_accum:
                    optimizer.zero_grad()
                    ib_count = 0
                ib_count += 1

                # Invoke the model
                model_outputs = self.call_model_fn(self.model, batch, trainer=self)

                # Evaluate loss
                loss = self.eval_loss(model_outputs, batch)

                # Get gradients
                loss.backward()
                
                # Step the optimizer
                if ib_count >= self.args.batch_accum:
                    optimizer.step()

                    # Step schedules
                    if lr_scheduler is not None:
                        lr_scheduler.step(self.curr_iter)

                # Evaluate
                if self.curr_iter % self.args.eval_every == 0 and self.curr_iter > 0:
                    self.evaluate_on_val()

                this_train_loss = loss.item()
                total_train_loss += this_train_loss
                total_train_loss_count += 1

                self.curr_iter += self.args.batch_size
               
                # Update states
                self.running_train_loss += this_train_loss
                self.running_train_loss_count += 1


            # Always evaluate at end of epoch
            epoch_loss = self.evaluate_on_val(viz=True)
            mean_train_loss = total_train_loss / total_train_loss_count
            print("\n")
            print("    epoch {} [it: {}]: eval loss = {}  train loss = {}".format(self.curr_epoch, self.curr_iter, epoch_loss.item(), mean_train_loss))
            print("        parameters: lr = {0:.10f}".format(lr_lbmd(self.curr_iter) * self.args.lr))

            self.save_training_state(opt=optimizer, suffix="epoch{:03d}".format(self.curr_epoch))

            self.curr_epoch += 1

    # Evaluate the loss for a batch, distributing over batches independently
    def eval_loss(self, model_outputs, batch, viz_extra=False):

        # Evaluate loss
        batch_loss = torch.tensor(0.0, device=world.device)
        batch_size = self.args.batch_size

        # Iterate through the batch, invoking the loss
        mean_batch_loss = self.loss_fn(batch, model_outputs, viz_extra=viz_extra, trainer=self)

        return mean_batch_loss

    # Add an entry to the statistic sets tracked during evaluation
    # value should be a plain float
    def add_eval_stat_entry(self, name, value):
        if name not in self.eval_stats:
            self.eval_stats[name] = []

        self.eval_stats[name].append(value)

    # Evaluate loss over (a subset of) the validation dataset and report results
    def evaluate_on_val(self, viz=False):
        self.model.eval()
        self.training = False
        self.eval_iter = self.curr_iter
        self.eval_stats = {} # clear it

        with torch.no_grad():

            total_loss = torch.tensor(0.0, device=world.device)

            eval_count = 0
            for batch_ind, batch in enumerate(self.val_loader):

                # Only evaluate on the first eval_size entries
                if batch_ind * self.args.batch_size >= self.args.eval_size:
                    break

                # Invoke the model
                model_outputs = self.call_model_fn(self.model, batch, trainer=self)

                # Evaluate loss
                total_loss += self.eval_loss(model_outputs, batch, viz_extra=viz)
                eval_count += 1
                self.eval_iter += self.args.batch_size

            total_loss /= eval_count

            world.tb_writer.add_scalar("evaluate loss", total_loss, self.curr_iter)
            print("    evaluation [it: {}]: loss = {}      train loss since last = {}".format(
                self.curr_iter, total_loss.item(), self.running_train_loss / (self.running_train_loss_count+1e-4)))

            # Print the mean of any tracked statistics
            for name in self.eval_stats:
                val = np.mean(self.eval_stats[name])
                print("      {} : {}".format(name, val))


            self.running_train_loss = 0.
            self.running_train_loss_count = 0

            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.save_training_state(suffix="best")

            self.save_training_state(suffix="last")

            return total_loss
