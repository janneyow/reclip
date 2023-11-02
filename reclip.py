from collections import defaultdict
import json
import argparse
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from interpreter import *
from executor import *
from methods import *

METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}

###
# THIS IS STILL A WIP - need to replace the args
###

if __name__ == "__main__":
    
    input_file = "reclip_data/test.jsonl"
    image_root = "reclip_data/images/"
    output_file = "output/reclip_test.txt"
    cache_path = None
    detector_file = None    # to include if want to specify object detections
                            # if not included, gold object boxes will be used
    
    device = 0            # CUDA device to use
    shuffle_words = False
    part = None

    method = "parse"
    
    box_method_aggregator = "sum"
    box_representation_method = "crop,blur"
    box_area_threshold = 0.0

    clip_model = "RN50x16,ViT-B/32"
    non_square_size = False
    expand_position_embedding = False
    blur_std_dev = 100

    # Parse settings
    baseline_threshold = 0.5
    temperature = 1.0
    superlative_head_only = False
    expand_chunks = False
    parse_no_branch = False
    possessive_expand = False
    no_possessive = False
    sigmoid = False

    # Heuristics settings
    no_rel = False
    no_sup = False
    no_null = False
    no_ternary = False

    # Load data from input_file
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    device = f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"

    # Load executor
    executor = ClipExecutor(clip_model=clip_model, 
                            box_representation_method=box_representation_method, 
                            method_aggregator=box_method_aggregator, 
                            device=device, 
                            square_size=not non_square_size, 
                            expand_position_embedding=expand_position_embedding, 
                            blur_std_dev=blur_std_dev, 
                            cache_path=cache_path)
    
    method = METHODS_MAP[method](args)
    output_file = open(output_file, "w")




