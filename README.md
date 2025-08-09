# Gulugod ng Bicolandia: Identifying Critical Corridors in the Bicol Transport Network for Typhoon Resilience

## Abstract

This research addresses the vulnerability of transportation network in the disaster-prone Bicol peninsula of the Philippines. The central question is how to identify the most critical infrastructure within this complex network to ensure the efficiency of relief operations during and after a catastrophic event. This study matters because it provides a data-driven methodology for prioritizing infrastructure investment and pre-positioning disaster response resources, moving beyond generalized risk assessments to a targeted, network-aware strategy. The approach involved constructing a comprehensive multi-modal transport graph of the region's 114 municipalities, integrating road and ferry data. We then simulated network disruptions by systematically removing nodes within 671 historical storm swaths, guided by centrality metrics (betweenness, degree, closeness) to measure the resulting decline in logistical efficiency. The analysis revealed that the network’s integrity is disproportionately dependent on a small subset of nodes with high betweenness centrality, which function as critical bridges. The targeted removal of just 1-2% of these nodes triggered a catastrophic failure in delivery efficiency, while the removal of other node types had a far less severe impact. These findings provide a crucial evidence base for policymakers, enabling a shift toward targeted infrastructure hardening. By identifying and mapping the specific corridors most vital to regional connectivity—primarily in the provinces of Camarines Sur and Albay—this research offers a precise framework for enhancing resilience and optimizing humanitarian logistics in the face of recurring natural disasters.

## Notebooks

Of course. Here is the summary of the codebase presented in a Markdown table.

| Notebook | Description |
|---|---|
| `01-boundaries.ipynb` | Downloads and filters Philippine administrative boundaries to isolate the Bicol Region, cleans town names, and saves the output as a GeoPackage file. |
| `02-transport-network.ipynb` | Creates a unified, multi-modal transportation network for the Bicol Region by downloading, merging, and simplifying road and ferry route data from OpenStreetMap. |
| `03-network-metrics.ipynb` | Analyzes the transport network's structure by calculating, normalizing, and mapping various node and edge centrality metrics, as well as detecting network communities. |
| `04-travel-times.ipynb` | Calculates and maps the shortest travel times from a central relief hub to all town centers by assigning weights based on road classification speed limits and distances. |
| `05-storm-swaths.ipynb` | Processes historical IBTrACS storm data to generate and map geographic "swaths" for cyclones that entered the Philippine Area of Responsibility and intersected the Bicol transport network. |
| `06-damage-simulation.ipynb` | Simulates the decline in network efficiency by systematically removing nodes within storm swaths based on various damage levels and centrality-based strategies. |
| `09-critical-infrastructure.ipynb` | Identifies, maps, and quantifies the distribution of critical transportation nodes—defined by top-percentile betweenness centrality—across the region's provinces and towns. |
