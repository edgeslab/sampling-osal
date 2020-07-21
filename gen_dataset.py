import pdb
import os
import copy
import json
import argparse
import numpy as np
import networkx as nx
from random import random

from lib.synthgen.SynthNetGen import SynthNetGen

DATA_SOURCE 		= "synth"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='', help='Name of config json to load')
    args = parser.parse_args()

    config_file = open(args.config)
    config = json.load(config_file)
    config_file.close()

    dataset_name = config["dataset"]["name"]
    num_graphs = config["dataset"]["num_graphs"]
    noise = config["dataset"]["noise"]
    random_seed = config["dataset"]["random_seed"]

    np.random.seed(random_seed)

    dataset_dir = os.path.join('data', dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    params = config["params"]

    def validate(attr, value):
        if attr == 'num_attributes':
            return int(np.floor(value))
        elif attr == 'class_balance':
            return np.clip(value, 0.05, 0.95)
        elif attr == 'density':
            return np.clip(value, 0.0001, 0.2)
        elif attr == 'homophily':
            return np.clip(value, 0.05, 0.95)


    cat_count = 0
    for cat in config["categories"]:
        # cat_dir_name = 'cat%02d_%s' % (cat_count, cat["name"])
        cat_dir_name = 'cat%02d' % (cat_count)
        category_dir = os.path.join(dataset_dir, cat_dir_name)
        if not os.path.exists(category_dir):
            os.mkdir(category_dir)

        other_params = cat["params"]
        for i in range(num_graphs):
            new_params = copy.deepcopy(params)

            for k in other_params:
                if k == 'num_attributes':
                    new_params[k] = other_params[k]
                    continue
                old_val = other_params[k]
                new_val = np.random.normal(old_val, old_val * noise, 1)[0]
                new_val = validate(k, new_val)
                new_params[k] = new_val
            
            new_params["random_seed"] = np.random.randint(random_seed)
            generator = SynthNetGen(**new_params)
            generator.generate()

            graph_dir = os.path.join(category_dir, 'graph%02d' % i)
            if not os.path.exists(graph_dir):
                os.mkdir(graph_dir)
            generator.save_csv(graph_dir)
            
            # print 'graph saved:', graph_dir

        cat_count += 1
        # print ''


if __name__ == '__main__':
    main()