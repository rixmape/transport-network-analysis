import multiprocessing as mp
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
np.random.seed(42)

# --- Setup ---
BASE_PATH = "data"
TRANSPORT_NETWORK_FILE = os.path.join(BASE_PATH, "merged_network_weighted.graphml")
CENTRALITY_FILE = os.path.join(BASE_PATH, "centrality_measures.csv")
TRAVEL_TIMES_FILE = os.path.join(BASE_PATH, "travel_times.csv")
SWATHS_DIR = os.path.join(BASE_PATH, "storm_swath_geometries")
RESULTS_DIR = os.path.join(BASE_PATH, "simulation_results_networkx")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STORMS = 100
DAMAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
REMOVAL_STRATEGIES = ["degree", "betweenness", "closeness", "random"]

# --- Worker Globals ---
worker_graph = None
worker_base_times = None
worker_origins = None
worker_destinations = None


# --- Helper Functions ---
def load_graph():
    G = ox.load_graphml(TRANSPORT_NETWORK_FILE)
    for _, _, data in G.edges(data=True):
        data["segment_time"] = float(data["segment_time"])
    return G


def init_worker(G, base_times, origins, destinations):
    global worker_graph, worker_base_times, worker_origins, worker_destinations
    worker_graph = G
    worker_base_times = base_times
    worker_origins = origins
    worker_destinations = destinations


def get_removable_nodes(ranked_nodes, damage_level):
    if not ranked_nodes:
        return []
    count = int((damage_level / 100) * len(ranked_nodes))
    return ranked_nodes[:count]


def compute_new_travel_times(graph, nodes_to_remove, origins, destinations):
    remaining_nodes = [n for n in graph.nodes if n not in nodes_to_remove]
    graph_damaged = graph.subgraph(remaining_nodes)
    travel_times = []
    for origin, destination in zip(origins, destinations):
        try:
            time = nx.dijkstra_path_length(graph_damaged, source=origin, target=destination, weight="segment_time")
        except nx.NetworkXNoPath:
            time = np.inf
        travel_times.append(time)
    return travel_times


def compute_efficiency(base_times, new_times):
    base_times_arr = np.asarray(base_times)
    new_times_arr = np.asarray(new_times)
    n = base_times_arr.size
    if n == 0:
        return 0.0
    ratio = np.zeros_like(base_times_arr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(base_times_arr, new_times_arr, out=ratio, where=new_times_arr != 0)
    log_term = np.log2(1 + ratio)
    return np.sum(log_term) / n


def run_storm_task(args):
    sid, ranked_nodes, strategy, level = args
    nodes_to_remove = get_removable_nodes(ranked_nodes, level)
    new_times = compute_new_travel_times(
        worker_graph,
        nodes_to_remove,
        worker_origins,
        worker_destinations,
    )
    efficiency = compute_efficiency(worker_base_times, new_times)
    return {
        "storm_id": sid,
        "node_removal_strategy": strategy,
        "damage_level": level,
        "delivery_efficiency": efficiency,
    }


def process_storm(storm_id, nodes_in_swaths, df_centrality, origins, destinations, pool):
    result_file = os.path.join(RESULTS_DIR, f"storm_{storm_id}.csv")
    if os.path.exists(result_file):
        print(f"✅ Storm {storm_id}: result exists. Skipping.")
        return

    swath_nodes = nodes_in_swaths[nodes_in_swaths["SID"] == storm_id].index.tolist()
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
            storm_id,
            ranked_nodes[strategy],
            strategy,
            level,
        )
        for strategy, level in product(REMOVAL_STRATEGIES, DAMAGE_LEVELS)
    ]

    results = list(
        tqdm(
            pool.imap_unordered(run_storm_task, task_args),
            total=len(task_args),
            desc=f"🌪️  Storm {storm_id}",
        )
    )

    df_result = pd.DataFrame(results)
    df_result.to_csv(result_file, index=False)


if __name__ == "__main__":
    print("🚀 Starting damage simulation...")
    print("📦 Loading data...")

    G = load_graph()

    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    swath_files = glob(os.path.join(SWATHS_DIR, "*.gpkg"))
    gdf_swaths = pd.concat([gpd.read_file(f) for f in swath_files], ignore_index=True)
    gdf_swaths = gdf_swaths.head(N_STORMS)
    nodes_in_swaths = gpd.sjoin(gdf_nodes, gdf_swaths, how="inner", predicate="within")

    df_centrality = pd.read_csv(CENTRALITY_FILE).set_index("osmid")
    df_travel_times = pd.read_csv(TRAVEL_TIMES_FILE)
    base_travel_times = df_travel_times["travel_time"].values
    origin_nodes = df_travel_times["origin_node"].values
    destination_nodes = df_travel_times["destination_node"].values

    init_args = (G, base_travel_times, origin_nodes, destination_nodes)

    with mp.Pool(initializer=init_worker, initargs=init_args) as pool:
        print(f"⚡ Running simulation for {len(gdf_swaths)} storms...")

        for storm in gdf_swaths.itertuples():
            process_storm(
                storm.SID,
                nodes_in_swaths,
                df_centrality,
                origin_nodes,
                destination_nodes,
                pool,
            )

    print("Simulation complete.")
