import json
import numpy as np
from collections import defaultdict

# Min-max for node params.
# Mappings for each node b/c each node has unique params.
MIN_MAX_SURFACE_SAMPLER_PARAMS = {
    "point_extents": [20.0, 100.0],
    "points_per_squared_meter": [0.0, 7.0]
}

MIN_MAX_SPATIAL_NOISE = {
    "random_offset": [100000.0, 100000.0],
    "transform": [0.0, 25.0]
}

MIN_MAX_DENSITY_FILTER = {
    "lower_bound": [0.3, 0.8],
    "upper_bound": [0.5, 1.0],
}

MIN_MAX_TRANSFORM_POINTS = {
    "offset_min": [0.0, 0.0],
    "offset_max": [0.0, 100.0],
    "rotation_min": [0.0, 0.0],
    "rotation_max": [0.0, 360.0],
    "scale_min": [0.5, 5.0],
    "scale_max": [1.0, 10.0]
}

# Map node types to their respective min-max dictionaries
NODE_TYPE_MIN_MAX_MAP = {
    "PCGSurfaceSamplerSettings": MIN_MAX_SURFACE_SAMPLER_PARAMS,
    "PCGSpatialNoiseSettings": MIN_MAX_SPATIAL_NOISE,
    "PCGDensityFilterSettings": MIN_MAX_DENSITY_FILTER,
    "PCGTransformPointsSettings": MIN_MAX_TRANSFORM_POINTS
}

def normalize_parameter(value, min_val, max_val):
    """Normalize a value to the range [0, 1]."""
    if max_val - min_val == 0:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def flatten_params(params, parent_key=''):
    """Flatten nested parameters into a numeric array."""
    if isinstance(params, (int, float)):
        params = {"value": params}

    # Extract numeric values recursively
    flattened_values = extract_numeric_values(params)

    # Handle empty input
    if not flattened_values:  # If no numeric values are found, default to a zero vector
        flattened_values = [0.0]

    return np.array(flattened_values)

def extract_numeric_values(data):
    """Recursively extract numeric values from a nested dictionary."""
    if isinstance(data, dict):
        values = []
        for v in data.values():
            values.extend(extract_numeric_values(v))
        return values
    elif isinstance(data, (int, float)):
        return [data]
    return []  # Ignore non-numeric types

def normalize_node_parameters(node_params, node_settings):
    """Normalize node parameters using predefined min-max ranges."""
    # Check if the node settings exists in the min-max map.
    if node_settings not in NODE_TYPE_MIN_MAX_MAP:
        print(f"Warning: Node settings '{node_settings}' not found in min-max ranges. Skipping normalization.")
        return [] # Return empty if node_settings is not found in the map.
    
    param_ranges = NODE_TYPE_MIN_MAX_MAP[node_settings]
    normalized_params = []
    
    for key, value in node_params.items():
        if key not in param_ranges:
            print(f"Warning: Parameter '{key}' not found in min-max ranges for '{node_settings}'. Skipping.")
            continue  # Skip parameters not in the predefined ranges

        # Ensure min-max range is a tuple(min_val, max_val).
        range_val = param_ranges[key]
        if isinstance(range_val, (int, float)): # If it is a single value, make it a tuple.
            min_val = max_val = range_val
        else:
            min_val, max_val = range_val # Unpack as a tuple.

        # Flatten the parameter values and normalize.
        flat_values = flatten_params(value)
        normalized_values = [
            min(max(round(normalize_parameter(v, min_val, max_val), 8), 0.0), 1.0)
            for v in flat_values
            ]
        normalized_params.extend(normalized_values)
    
    return normalized_params

def process_pcg_data(data):
    """Process PCG data with normalization using predefined min-max ranges."""
    processed_data = {"General": {}, "Undesirable": {}} # Ensure both categories are initialized.

    for category, graphs in data.items():
        # Iterate over "General" and "Undesirable" categories.
        for graph_name, graph_data in graphs.items():
            processed_data[category][graph_name] = {}

            if "types" not in graph_data:
                print(f"Warning: Graph '{graph_name}' in category '{category}' has no 'types' key. Skipping.")
                continue

            for type_name, type_data in graph_data["types"].items():
                # Ensure the type_data contains a "Nodes" key with a list of dictionaries.
                if "Nodes" not in type_data or not isinstance(type_data["Nodes"], list):
                    print(f"Warning: Type '{type_name}' in graph '{graph_name}' has no valid 'Nodes'. Skipping.")
                    continue
                
                for node in type_data["Nodes"]:
                    if not isinstance(node, dict):
                        print(f"Warning: invalid node format in '{type_name}' of '{graph_name}': '{node}'")
                        continue

                    node_settings = node.get("node_settings")
                    if not node_settings:
                        print(f"Warning: Node '{node.get('node_name', 'Unknown')}' has no 'node_settings'. Skipping.")
                        continue

                    if node_settings not in NODE_TYPE_MIN_MAX_MAP:
                        print(f"Warning: Unmapped node setting '{node_settings}' in graph '{graph_name}', skipping.")
                        continue

                    node_params = node.get("parameters", {})
                    normalized_params = normalize_node_parameters(node_params, node_settings)

                    if type_name not in processed_data[category][graph_name]:
                        processed_data[category][graph_name][type_name] = []

                    processed_data[category][graph_name][type_name].append(normalized_param)

    return processed_data

def remove_empty_nodes(dataset):
    """Remove empty nodes from the dataset."""
    for category, graphs in dataset.items():
        for graph_name, graph_data in graphs.items():
            for type_name, nodes in graph_data.items():
                if isinstance(nodes, list):
                    # Remove nodes that are empty
                    category[:] = [node for node in category if node and len(node) > 0]
                else:
                    print(f"Warning: Skipping invalid nodes in '{type_name}' of '{graph_name}': '{nodes}'")
    return dataset

def pad_feature_vectors(dataset):
    """Pad all feature vectors to the same length."""
    # Ensure dataset is not empty.
    if not dataset:
        raise ValueError("The dataset is empty. Ensure valid data before calling this function.")

    # Calculate maximum length across all nodes.
    max_len = 0
    for category, graphs in dataset.items():
        for graph_name, graph_data in graphs.items():
            for type_name, nodes in graph_data.items():
                if isinstance(nodes, list):
                    for node in nodes:
                        if isinstance(node, list):
                            max_len = max(max_len, len(node))

    if max_len == 0:
        raise ValueError("All nodes are empty after preprocessing. Ensure valid feature vectors are present.")

    print(f"Padding all vectors to length: {max_len}")

    # Pad each node to the maximum length.
    for category, graphs in dataset.items():
        for graph_name, graph_data in graphs.items():
            for type_name, nodes in graph_data.items():
                if isinstance(nodes, list):
                    for i, node in enumerate(category):
                        if isinstance(node, list):
                            if len(node) < max_len:
                                category[i] = node + [0.0] * (max_len - len(node))
                        elif not node:
                            category[i] = [0.0] * max_len # Replace empty nodes with zeroes.
                        else:
                            print(f"Error: non-list node encountered during padding. Node: {node}.")
                            raise TypeError(f"Expected list, got {type(node)}")
    return dataset

def debug_dataset(dataset, message):
    """Debugging function to inspect dataset structure."""
    print(f"\nDEBUG: {message}")
    if not dataset:
        print("Dataset is empty.")
        return
    
    print(f"Top-level categories: {', '.join(dataset.keys())}.")
    for category, graphs in dataset.items():
        print(f"\nCategory: {category}, Number of graphs: {len(graphs)}")
        for graph_name, graph_data in graphs.items():
            print(f"    Graph: '{graph_name}'")
            for type_name, nodes in graph_data.items():
                if not isinstance(nodes, list): # Skip invalid entries.
                    print(f"    Types: {type_name}, Value: {nodes} (Not a list, skipping!)")
                    continue
                print(f"    Types: {type_name}, Number of Nodes: {len(nodes)}")
                for i, node in enumerate(nodes[:3]): # Print first three nodes for inspection.
                    print(f"    Node {i + 1}: {node}")

# Load JSON data.
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Save processed data.
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Preprocessed data saved to {filename}")

# Main Processing
input_file = "all_graphs_data.json"
output_file = "pcg_preprocessed_global_normalized.json"

# Step 1: Load raw data.
pcg_data = load_json(input_file)
debug_dataset(pcg_data, "Loaded Raw Data")

# Step 2: Normalize parameters
normalized_data = process_pcg_data(pcg_data)
debug_dataset(normalized_data, "After Normalization")

# Step 3: Remove empty nodes
cleaned_data = remove_empty_nodes(normalized_data)
debug_dataset(cleaned_data, "After removing empty nodes")

# Step 4: Pad feature vectors
try:
    padded_data = pad_feature_vectors(cleaned_data)
    debug_dataset(padded_data, "After padding feature vectors")
except Exception as e:
    print(f"Error during padding: {e}")
    exit(1)

# Step 5: Save processed data
save_to_json(padded_data, output_file) 