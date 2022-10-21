import argparse
import json
import os

def save_args(save_path, args: argparse.Namespace):
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

def load_args(load_path):
    args = argparse.Namespace()
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args