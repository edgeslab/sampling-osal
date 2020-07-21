import os, sys, pdb, math, json, operator, logging
import random, argparse

from sys import platform
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from sklearn.model_selection import train_test_split

from lib.graph.GraphData import GraphData
from algo.algo_factory import AlgoFactory


#####################################################################
# ------------------------ caching functions ---------------------- #
#####################################################################

def get_cache_filename(cache_dir, t, clf, eid, algo):
    tokens = eid.split('_')
    eid = '%s_%s' % (tokens[0], '_'.join(tokens[2:]))
    return cache_dir + '/t%d_%s_%s_%s.csv' % (t, eid, clf, algo)

def store_cache_trial(filename, xvals, yvals, times):
    dump = {
        'xvals' :   xvals,
        'yvals' :   yvals,
        'times' :   times
    }
    df = pd.DataFrame(dump)
    df.to_csv(filename, index=None)

def load_cache_trial(filename):
    return pd.read_csv(filename)





#####################################################################
# --------------------- result dumping functions ------------------- #
#####################################################################


def dump_results_to_csv(xlabel, xvals, yvals, exp_id, name):
    result = pd.DataFrame()
    result[xlabel] = xvals

    for data, algo in yvals:
        result[ '%s_%s' % (name, algo)] = data

    filename = "%s_%s.csv" % (name, exp_id)
    result.to_csv("results/" + filename, index=None)


def dump_timing_results_to_csv(xlabel, xvals, yvals, exp_id):
    result = pd.DataFrame()

    for data, algo in xvals:
        result["time_" + algo] = data

    for data, algo in yvals:
        result["acc_" + algo] = data

    filename = "%s.csv" % exp_id
    result.to_csv("results/" + filename, index=None)





#####################################################################
# ---------------------------  experiment  ------------------------ #
#####################################################################

def run_experiment(data, config, cached=False, cache_dir=''):
    nodes = list(data.graph.nodes())
    y_true = list(map(lambda x: data.graph.node[x]['label'], nodes))

    x_axis = None
    algo_obj, avg_acc, timing = {}, {}, {}
    plot_points = int(config['al_budget'] / config['budget_step'])

    for algo in config["algos"]:
        algo_obj[algo] = AlgoFactory.get_algo(data, config['classifier'], algo)
        avg_acc[algo] = np.zeros((config["num_trials"], plot_points))
        timing[algo] = np.zeros((config["num_trials"], plot_points))

    for t in range(config["num_trials"]):
        if cached:
            for algo in config["algos"]:
                cache_file = get_cache_filename(cache_dir, t, config['classifier'], config["exp_id"], algo)
                df_trial = load_cache_trial(cache_file)
                if x_axis is None:
                    x_axis = df_trial['xvals']
                avg_acc[algo][t, :] = df_trial['yvals']
                timing[algo][t, :] = df_trial['times']
            continue

        # if not cached, run the actual exp
        train, test = train_test_split(nodes, train_size=0.8, test_size=None, random_state=config["seed"] + t, stratify=y_true)	#, random_state=t

        for algo in config["algos"]:
            random.seed(config["seed"] + t)
            np.random.seed(config["seed"] + t)
            tf.set_random_seed(config["seed"] + t)

            # feat_indices = np.random.choice(range(data.num_feat), feat_thresh, replace=False)
            algo_obj[algo].cap_features(range(data.num_feat))
            acc, times, stats = algo_obj[algo].execute(config, train, test)

            x_axis = acc[0] if x_axis is None else x_axis
            avg_acc[algo][t, :] += acc[1][-plot_points:]
            timing[algo][t, :] += times
            
            cache_file = get_cache_filename(cache_dir, t, config['classifier'], config["exp_id"], algo)
            store_cache_trial(filename=get_cache_filename(cache_dir, t, config['classifier'], config["exp_id"], algo), xvals=x_axis, yvals=acc[1][-plot_points:], times=times)

    y_vals = []
    y_er_vals = []
    t_vals = []
    t_er_vals = []

    for algo in config["algos"]:
        y = np.mean(avg_acc[algo], axis=0)
        y_error = np.std(avg_acc[algo], axis=0)

        y_vals.append((y[-plot_points:], algo))
        y_er_vals.append((y_error[-plot_points:], algo))

        t = np.mean(timing[algo], axis=0)
        t_error = np.std(timing[algo], axis=0)

        t_vals.append((t[-plot_points:], algo))
        t_er_vals.append((t_error[-plot_points:], algo))

    return (x_axis[-plot_points:], y_vals, y_er_vals, t_vals, t_er_vals)




#####################################################################
# ---------------------------  experiment  ------------------------ #
#####################################################################


def load_data(base_config, data_source):
    data_source_path	= os.path.join(base_config["root_dir"], data_source.replace('.', '/'))
    node_file 			= os.path.join(data_source_path, base_config["node_file"])
    edge_file 			= os.path.join(data_source_path, base_config["edge_file"])
    feature_file		= os.path.join(data_source_path, base_config["feature_file"])

    data = GraphData(node_file=node_file, edge_file=edge_file, feature_file=feature_file)

    # logging.info('clustering coeff: %0.4f' % nx.average_clustering(data.graph))
    # logging.info('Num triangles: %d' % np.count_nonzero(nx.triangles(data.graph).values()))
    # logging.info('Diameter: %d' % nx.diameter(data.graph))

    return data


def prepare_logger(exp_id):
    if platform == 'darwin':
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename='logs/'+exp_id+'.log', filemode='w', format='%(levelname)s:%(message)s', level=logging.DEBUG)

    logging.getLogger('matplotlib.font_manager').disabled = True
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=2162019, help='Random seed.')
    parser.add_argument('-b', type=int, default=-1, help='Maximum node budget for AL.')
    parser.add_argument('-nt', type=int, default=-1, help='Number of trials.')
    parser.add_argument('-algos', nargs='+', help='algorithms to run')
    parser.add_argument('-base', default='configs/base.json', help='Base config.')
    parser.add_argument('-config', default='configs/cc_citeseer.json', help='Experiment config.')
    parser.add_argument("--cached", action='store_true', help="load trial results from cache?", required=False)
    args = parser.parse_args()

    def update_configs(config):
        config["seed"] = args.s if args.s != -1 else config["seed"]
        config["num_trials"] = args.nt if args.nt != -1 else config["num_trials"]
        config["al_budget"] = args.b if args.b != -1 else config["al_budget"]
        config["algos"] = args.algos if args.algos is not None else config["algos"]

    base_config = json.loads(open(args.base, 'r').read())
    config = json.loads(open(args.config, 'r').read())
    update_configs(config)

    assert(config["al_budget"] % config["budget_step"] == 0)
    assert(config["budget_step"] % config["al_batch"] == 0)

    exp_id = "%s_%d_%d_%d_%d" % (config["dataset"], config["num_trials"], config["al_budget"], config["budget_step"], config["al_batch"])
    
    config['exp_id'] = exp_id
    prepare_logger(exp_id)
    data = load_data(base_config, config["dataset"])
    logging.info("Experiment Setup: %s" % (args))

    xvals, yvals, yvals_err, times, times_err = run_experiment(data, config, cached=args.cached, cache_dir=base_config['cache_dir'])

    dump_results_to_csv("Budget", xvals, yvals, exp_id, config["classifier"])
    dump_results_to_csv("Budget", xvals, yvals_err, exp_id + '_error', config["classifier"])
    dump_results_to_csv("Budget", xvals, times, exp_id + '_timing', config["classifier"])
    dump_results_to_csv("Budget", xvals, times_err, exp_id + '_timing_error', config["classifier"])

    logging.info("Done: %s" % exp_id)


if __name__ == "__main__":
    main()