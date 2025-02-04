import json

# Parameter map for node reconstruction.
PARAMETER_MAP = {
    "PCGSurfaceSamplerSettings": {
        "point_extents": ["x", "y", "z"],
        "points_per_squared_meter": None
    },
    "PCGSpatialNoiseSettings": {
        "random_offset": ["x", "y", "z"],
        "transform": {
            "translation": ["x", "y", "z"],
            "rotation": ["x", "y", "z", "w"],
            "scale": ["x", "y", "z"]
        }
    },
    "PCGTransformPointsSettings": {
        "offset_min": ["x", "y", "z"],
        "offset_max": ["x", "y", "z"],
        "rotation_min": ["pitch", "yaw", "roll"],
        "rotation_max": ["pitch", "yaw", "roll"],
        "scale_min": ["x", "y", "z"],
        "scale_max": ["x", "y", "z"]
    },
    "PCGDensityFilterSettings": {
        "lower_bound": None,
        "upper_bound": None,
        "invert_filter": None
    }
}

# Example parameter ranges (replace these with actual ranges used during preprocessing).
param_ranges = {
    "point_extents": (20, 100),
    "points_per_squared_meter": (0, 7),
    "random_offset": (0, 100000),
    "transform_translation": (0, 100000),
    "transform_rotation": (0, 360),
    "transform_scale": (0.5, 50),
    "offset_min": (0, 0),
    "offset_max": (0, 100),
    "rotation_min": (0, 0),
    "rotation_max": (0, 360),
    "scale_min": (0.5, 5),
    "scale_max": (1, 10),
    "lower_bound": (0.3, 0.8),
    "upper_bound": (0.5, 1),
    "invert_filter": (0, 1)
}

# Function to reverse normalization.
def reverse_normalize_parameters(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val

# Reconstruct the nested parameter structure.
def reconstruct_nested_params(param_name, values, param_def):
    """Reconstruct nested parameters (e.g. vectors or nested dictionaries)."""
    if isinstance(param_def, list):
        # Reconstruct a vector (e.g. (x, y, z)).
        if len(values) < len(param_def):
            raise IndexError(f"Not enough values for {param_name}, expected {len(param_def)}, got {len(values)}.")
        return {key: reverse_normalize_parameters(value, *param_ranges[param_name]) for key, value in zip(param_def, values)}

    elif isinstance(param_def, dict):
        # Reconstruct a nested dictionary.
        result = {}
        for key, sub_keys in param_def.items():
            count = len(sub_keys) else 1
            # Ensure that we do not go out of range when there are fewer values left.
            if len(values) < count:
                raise IndexError(f"Not enough value for nested parameter {key}, expected {count}, got {len(values)}.")
            result[key] = reconstruct_nested_params(f"{param_name}_{key}", values[:count], sub_keys)
            values = values[count]
        return result

    elif param_def is None:
        # Scalar praameter (no list or dict).
        if len(values) < 1:
            raise IndexError(f"Not enough values for scalar parameter {param_name}")
        return reverse_normalize_parameters(values[0], *param_ranges[param_name])

    else:
        # This should never happen if the input is valid.
        raise ValueError(f"Unexpected parameter definition: {param_def}")

# Reconstruct the graph from file data.
def reconstruct_graph_from_file(input_file, output_file, new_graph_name="ReconstructedGraph"):
    """
    Reconstruct a PCG graph from normalized data stored in a JSON file.

    Args:
        input_file: Path to the input JSON file containing normalized data.
        output_file: Path to save the reconstructed graph.
        new_graph_name: Name for the reconstructed graph.

    Returns:
        None. Saves the reconstructed graph to the output file.
    """
    with open(input_file, "r") as file:
        graph_data = json.load()
    
    reconstructed_graph = {"graph_name": new_graph_name, "types": {}}

    # Iterate through categories in the input data.
    for category_name, nodes in graph_data.items():
        reconstructed_graph["types"][category_name] = {"Meshes": [], "Nodes": []}

        # Map each node_vector to a specific node type.
        for node_type, param_defs in PARAMETER_MAP.items():
            if not nodes:
                continue # Skip if there is no data for this category.
            
            # Each node_vector corresponds to a node in PARAMETER_MAP.
            node_vector = nodes.pop(0) # Get the next available node vector.
            reconstructed_node = {
                "node_name": f"{node_type}_Node",
                "node_settings": node_type,
                "parameters": {}
            }

            # Reconstruct parameters for the node.
            for param_name, param_def in param_defs.items():
                param_size = len(param_def) if isinstance(param_def, list) else (len(param_def) if isinstance(param_def, dict) else 1)
                values = node_vector[:param_size] # Extract the required number of values.
                try:
                    reconstructed_node["parameters"][param_name] = reconstruct_nested_params(param_name, values, param_def)
                except IndexError as e:
                    print(f"Error processing {param_name}: {e}.")
                    reconstructed_node["parameters"][param_name] = None # Set to None or default.
                node_vector = node_vector[param_size:] # Consume used values.

            reconstructed_graph["types"][category_name]["Nodes"].append(reconstructed_node)
    
    # Save the reconstructed graph.
    with open(output_file, "w") as output_file:
        json.dump(reconstructed_graph, output_file, indent=4)
    print(f"Reconstructed graph saved to {output_file.name}.")

input_file = "generated_pcg_graph.json"
output_file = "reconstructed_graph.json"

# Reconstruct and save the graph.
reconstruct_graph_from_file(input_file, output_file)