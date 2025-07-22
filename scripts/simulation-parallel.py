import multiprocessing
import os
import warnings
from glob import glob
from itertools import product

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_PATH = "data"
TRANSPORT_NETWORK_FILE = os.path.join(BASE_PATH, "merged_network_weighted.graphml")
CENTRALITY_FILE = os.path.join(BASE_PATH, "centrality_measures.csv")
TRAVEL_TIMES_FILE = os.path.join(BASE_PATH, "travel_times.csv")
SWATHS_DIR = os.path.join(BASE_PATH, "storm_swath_geometries")

RESULTS_DIR = os.path.join(BASE_PATH, "simulation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DAMAGE_LEVELS = [0.12, 0.25, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 10.00, 20.00, 30.00, 40.00, 50.00]
REMOVAL_STRATEGIES = ["degree", "betweenness", "closeness", "random"]

worker_base_times = None
worker_origins = None
worker_destinations = None


def init_worker(graph, base_times, origins, destinations):
    """Initializer for each worker process."""
    global worker_graph, worker_base_times, worker_origins, worker_destinations
    worker_graph = graph
    worker_base_times = base_times
    worker_origins = origins
    worker_destinations = destinations


def get_removable_nodes(ranked_nodes, damage_level):
    """Calculates which nodes to remove based on a damage percentage."""
    if not ranked_nodes:
        return []
    count = int((damage_level / 100) * len(ranked_nodes))
    return ranked_nodes[:count]


def compute_new_travel_times(graph, nodes_to_remove, origins, destinations):
    """Calculates new travel times on a damaged graph."""
    graph_damaged = graph.subgraph([n for n in graph.nodes if n not in nodes_to_remove])
    times = []
    for origin, destination in zip(origins, destinations):
        try:
            time = nx.dijkstra_path_length(graph_damaged, source=origin, target=destination, weight="segment_time")
        except nx.NetworkXNoPath:
            time = np.inf
        times.append(time)
    return times


def compute_efficiency(base_times, new_times):
    """Computes the network efficiency metric."""
    n = len(base_times)
    if n == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.array(base_times) / np.array(new_times)
        ratio = np.nan_to_num(ratio, nan=0.0, neginf=0.0)
    log_term = np.log2(1 + ratio)
    return np.sum(log_term) / n


def run_storm_task(args):
    """The core task executed by each worker process."""
    sid, ranked_nodes, strategy, level = args
    nodes_to_remove = get_removable_nodes(ranked_nodes, level)
    new_times = compute_new_travel_times(worker_graph, nodes_to_remove, worker_origins, worker_destinations)
    efficiency = compute_efficiency(worker_base_times, new_times)
    return {
        "storm_id": sid,
        "node_removal_strategy": strategy,
        "damage_level": level,
        "delivery_efficiency": efficiency,
    }


def process_storm(storm_data, graph, nodes_in_swaths, df_centrality, base_times, origins, destinations):
    """Processes all simulation scenarios for a single storm."""
    sid = storm_data.SID
    result_file = os.path.join(RESULTS_DIR, f"storm_{sid}.csv")
    if os.path.exists(result_file):
        print(f"✅ Storm {sid}: result file exists. Skipping.")
        return

    swath_nodes = nodes_in_swaths[nodes_in_swaths["SID"] == sid].index.tolist()
    protected_nodes = set(origins) | set(destinations)
    removable_nodes = [node for node in swath_nodes if node not in protected_nodes]
    ranked_nodes = {}

    if removable_nodes:
        centrality_nodes = df_centrality.loc[removable_nodes]
        for strategy in ["degree", "betweenness", "closeness"]:
            ranked_nodes[strategy] = centrality_nodes[strategy].sort_values(ascending=False).index.tolist()
        ranked_nodes["random"] = list(np.random.permutation(removable_nodes))
    else:
        for strategy in REMOVAL_STRATEGIES:
            ranked_nodes[strategy] = []

    task_args = [
        (
            sid,
            ranked_nodes[strategy],
            strategy,
            level,
        )
        for strategy, level in product(REMOVAL_STRATEGIES, DAMAGE_LEVELS)
    ]

    init_args = (graph, base_times, origins, destinations)

    with multiprocessing.Pool(initializer=init_worker, initargs=init_args) as pool:
        results_iterator = pool.imap_unordered(run_storm_task, task_args)
        results = []
        for result in tqdm(results_iterator, total=len(task_args), desc=f"🌪️ Processing Storm {sid}"):
            results.append(result)

    df_result = pd.DataFrame(results)
    df_result.to_csv(result_file, index=False)


if __name__ == "__main__":
    print("Starting simulation process...")
    print("Loading data...")

    graph_transport = ox.load_graphml(TRANSPORT_NETWORK_FILE)
    for _, _, data in graph_transport.edges(data=True):
        data["segment_time"] = float(data["segment_time"])

    gdf_nodes = ox.graph_to_gdfs(graph_transport, edges=False)
    gdf_swaths = pd.concat([gpd.read_file(f) for f in glob(os.path.join(SWATHS_DIR, "*.gpkg"))], ignore_index=True)
    gdf_swaths = gdf_swaths.head(100)
    nodes_in_swaths = gpd.sjoin(gdf_nodes, gdf_swaths, how="inner", predicate="within")

    df_centrality = pd.read_csv(CENTRALITY_FILE).set_index("osmid")
    df_travel_times = pd.read_csv(TRAVEL_TIMES_FILE)
    base_travel_times = df_travel_times["travel_time"].values
    origin_nodes = df_travel_times["origin_node"].values
    destination_nodes = df_travel_times["destination_node"].values

    print(f"Processing {len(gdf_swaths)} storms...")

    for storm in gdf_swaths.itertuples():
        process_storm(
            storm,
            graph_transport,
            nodes_in_swaths,
            df_centrality,
            base_travel_times,
            origin_nodes,
            destination_nodes,
        )

    print("Simulation complete.")
