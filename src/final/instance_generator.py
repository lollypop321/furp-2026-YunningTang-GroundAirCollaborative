# ===== instance_generator.py =====
import pickle
import os
import numpy as np
import random
from config import SEEDS, SCALE_CONFIGS

def build_instance(cust_n, coords_scale, e_window, l_window):
    """
    Generate a single TDRPTW instance.

    Parameters:
        cust_n (int): Number of customers
        coords_scale (int): Range for random customer coordinates [-scale, scale]
        e_window (list): [min, max] for earliest service time
        l_window (list): [min, max] for latest service time

    Returns:
        nodes (np.ndarray): Coordinates of depot + customers
        dist (np.ndarray): Euclidean distance matrix
        demand (np.ndarray): Random demand for each customer
        e (np.ndarray): Earliest service time for each customer
        l (np.ndarray): Latest service time for each customer
    """
    depot = np.array([[0, 0]])
    custs = np.random.uniform(-coords_scale, coords_scale, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    n = len(nodes)
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(e_window[0], e_window[1], cust_n)
    l = np.random.randint(l_window[0], l_window[1], cust_n)
    return nodes, dist, demand, e, l

def generate_all_instances():
    """
    Generate all instances for all scales and random seeds.
    Instances are saved as pickle files in ../data/instances/
    """
    save_dir = "../data/instances"
    os.makedirs(save_dir, exist_ok=True)
    
    for scale, config in SCALE_CONFIGS.items():
        for seed in SEEDS:
            # Set random seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Generate instance with scale-specific parameters
            nodes, dist_mat, demand_list, e_list, l_list = build_instance(
                scale,
                coords_scale=config['coords_scale'],
                e_window=config['e_window'],
                l_window=config['l_window']
            )

            # Package instance data
            instance = {
                'scale': scale,
                'seed': seed,
                'nodes': nodes,
                'dist_mat': dist_mat,
                'demand_list': demand_list,
                'e_list': e_list,
                'l_list': l_list,
                'config': config,
            }

            # Save to pickle file
            filename = f"instance_{scale}_{seed}.pkl"
            with open(os.path.join(save_dir, filename), 'wb') as f:
                pickle.dump(instance, f)
            
            print(f"Generated: {filename}")

if __name__ == "__main__":
    generate_all_instances()