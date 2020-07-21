import sys
import torch

# Globals and other world state management


# Global argument cache
global args
args = []

# Global torch default device
global device
device = None

global dtype
dtype = None

# Debug checks
global debug_checks
debug_checks = False

# Tensorboard logger
global tb_writer
tb_writer = None
global tb_tick
tb_tick = 0

global train_state
train_state = None


class ArgsObject(object):
    pass

def args_to_str(args):

    s = []

    for attr, value in args.__dict__.items():
        if(attr != ""):
            s.append(attr + ": " + str(value))

    return "\n".join(s)
