import unreal
import os
import json
import re
import traceback

try:
    from name_parser import MeshNameParser  # or from mesh_name_parser if you've renamed it
    print("Import class MeshMameParser successful!")
except Exception as e:
    print("ImportError:", str(e))

try:
    from tag_node import TagSpawnerNode  # or from mesh_name_parser if you've renamed it
    print("Import class TagSpawnerNode successful!")
except Exception as e:
    print("ImportError:", str(e))

# Define a dictionary (hashmap) where keys are node types and values are lists of relevant parameters
PARAMETER_MAP = {
    "PCGSurfaceSamplerSettings": ["point_extents", "points_per_squared_meter"],
    "PCGSpatialNoiseSettings": ["random_offset", "transform"],
    "PCGDensityFilterSettings": ["lower_bound", "upper_bound", "invert_filter"],
    "PCGTransformPointsSettings": ["offset_min", "offset_max", "rotation_min", "rotation_max", "scale_min", "scale_max"]
}

# Obtain selection of PCG graphs.
def get_all_pcg_graphs():
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    # Filter for PCGGraph assets
    all_assets = asset_registry.get_assets_by_path("/Game", True)
    
    # Regular expression to match asset names like "PCG_Env1", "PCG_Env2", ..., "PCG_Env30"
    pattern = re.compile(r"^PCG_Env\d+$")

    # Regular expression to match asset names starting with 'UNDESIRABLE_'
    undesirable_pattern = re.compile(r"^UNDESIRABLE_")
    
    # Filter only PCGGraph assets using the full class path, and check asset name matches the pattern
    filtered_pcg_assets = [
        asset for asset in all_assets
        if "PCGGraph" in str(asset.asset_class_path) and (
            pattern.match(str(asset.asset_name)) or undesirable_pattern.match(str(asset.asset_name))
        )
    ]
    
    pcg_graphs = []
    for asset in filtered_pcg_assets:
        try:
            # Load the asset using get_asset()
            pcg_graph = asset.get_asset()
            if pcg_graph:
                pcg_graphs.append(pcg_graph)
        except Exception as e:
            unreal.log_error(f"Error loading asset {asset.object_path}: {str(e)}")
    
    return pcg_graphs

def clean_name(name):
    # Removes the trailing underscore and number from the node name (e.g., "StaticMeshSpawner_5" -> "StaticMeshSpawner").
    cleaned_name = re.sub(r'_\d+$', '', name)
    return cleaned_name

# Parse static mesh spawner node for information:
def parse_static_mesh_node(node, parser_instance):
    settings = node.get_settings()

    mesh_selector_params = settings.get_editor_property("mesh_selector_parameters")
    if isinstance(mesh_selector_params, unreal.PCGMeshSelectorWeighted):
        print("Mesh selector is weighted.")
        print("\n--- Mesh Selector Weighted Details ---")
        
        mesh_entries = mesh_selector_params.get_editor_property("mesh_entries")
        if mesh_entries:
            for mesh_entry in mesh_entries:
                print("Mesh entry type: ", type(mesh_entry).__name__)

                descriptor = mesh_entry.get_editor_property("descriptor")
                weight = mesh_entry.get_editor_property("weight")

                print("Weight: ", weight)

                if descriptor:
                    # Can we access static mesh of descriptor?
                    static_mesh_found = descriptor.get_editor_property("static_mesh")
                    
                    if static_mesh_found:
                        static_mesh_name = static_mesh_found.get_name()
                        print(f"  Static Mesh: {static_mesh_name}")

                        cleaned_static_mesh_name = clean_name(static_mesh_name)
                        print(f"Cleaned Mesh Name: {cleaned_static_mesh_name}")

                        parser_instance.associate_name_with_keyword(cleaned_static_mesh_name)

# Serialise function for more abstract UE datatypes.
def serialize_value(value):
    """
    Converts Unreal Engine-specific types (e.g., Vector) into JSON-serializable formats.
    """
    if isinstance(value, unreal.Vector):
        return {"x": value.x, "y": value.y, "z": value.z}
    elif isinstance(value, unreal.Color):
        return {"r": value.r, "g": value.g, "b": value.b, "a": value.a}
    elif isinstance(value, unreal.Rotator):
        return {"pitch": value.pitch, "yaw": value.yaw, "roll": value.roll}
    elif isinstance(value, unreal.Transform):
        return {
            "translation": serialize_value(value.translation),
            "rotation": serialize_value(value.rotation),
            "scale": serialize_value(value.scale3d)
        }
    elif isinstance(value, unreal.Quat):  # Serialize Quaternion
        return {"x": value.x, "y": value.y, "z": value.z, "w": value.w}
    elif isinstance(value, list):  # Handle lists recursively
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):  # Handle dicts recursively
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        return value  # Default case for standard JSON-serializable types

# Extract node data. To run on each PCG node parsed.
def extract_node_data(node):
    try:
        node_data = {
            "node_name": clean_name(node.get_name()),
            "node_settings": type(node.get_settings()).__name__,
            "parameters": extract_node_parameters(node)
        }

        return node_data
    except Exception as e:
        unreal.log_error(f"Error extracting data from node {node.get_name()}: {str(e)}")
        return None
    
# Extract node parametres based on respective mapping in PARAMETER_MAP.
def extract_node_parameters(node):
    node_parameters = {}
    node_settings_name = type(node.get_settings()).__name__

    node_settings = node.get_settings()

    if node_settings_name in PARAMETER_MAP:
        # Retrieve associated list of parameters.
        parameters_to_extract = PARAMETER_MAP[node_settings_name]

        for param_name in parameters_to_extract:
            try:
                param_value = node_settings.get_editor_property(param_name)
                node_parameters[param_name] = serialize_value(param_value)
            except Exception as e:
                unreal.log_warning(f"Could not extract {param_name} from {node_settings_name}: {str(e)}")
    
    return node_parameters

def extract_pcg_graph_data(pcg_graph, parser_instance, node_tag_instance):
    graph_data = {
        "graph_name": pcg_graph.get_name(),
        # Add category here from parser_instance
        "types": {}
    }
    
    for type_name, meshes in parser_instance.TYPE_MAP.items():
        graph_data["types"][type_name] = {
            "Meshes": meshes,
            "Nodes": []  # Add nodes here
        }

    # Tag node logic. Comes before param finding and filling.
    spawner_pos_instance = {}
    grouped_nodes_instance = {}

    node_tag_instance.fill_spawner_nodes(pcg_graph, spawner_pos_instance, grouped_nodes_instance)

    nodes = pcg_graph.nodes
    y_threshold = 20

    # Before loading up params, group nodes by association.
    for node in nodes:

        close_nodes_to_spawner = node_tag_instance.check_node_proximity_to_static_mesh_spawner(node, spawner_pos_instance, y_threshold)
        for spawner_name in close_nodes_to_spawner:
            if spawner_name in grouped_nodes_instance:
                grouped_nodes_instance[spawner_name].append(node.get_name())
            else:
                print(f"Warning: Static Mesh Spawner '{spawner_name}' not found in grouped_nodes dictionary.")

    # Test for node tagging.
    try:
        print("\n--- Grouped Nodes by Static Mesh Spawner ---")
        for spawner_name, grouped_nodes in grouped_nodes_instance.items():
            print(f"Static Mesh Spawner '{spawner_name}' has the following grouped nodes:")
            for grouped_node in grouped_nodes:
                print(f"  - {grouped_node}")
        print("--- End of Grouped Nodes ---")
    except Exception as e:
        print(f"Error printing grouped nodes: {e}")

    for spawner_name, grouped_nodes in grouped_nodes_instance.items():
        # Get current node from list of nodes.
        spawner_node = next((n for n in nodes if n.get_name() == spawner_name), None)
        if not spawner_node or not isinstance(spawner_node.get_settings(), unreal.PCGStaticMeshSpawnerSettings):
            print(f"Warning: '{spawner_name}' is not a valid StaticMeshSpawner node.")
            continue

        settings = spawner_node.get_settings()
        mesh_selector_params = settings.get_editor_property("mesh_selector_parameters")

        # Parse static mesh.
        parse_static_mesh_node(spawner_node, parser_instance)

        # Check for static meshes in the node and categorize them
        if isinstance(mesh_selector_params, unreal.PCGMeshSelectorWeighted):
            mesh_entries = mesh_selector_params.get_editor_property("mesh_entries")
            if mesh_entries:
                for mesh_entry in mesh_entries:
                    descriptor = mesh_entry.get_editor_property("descriptor")
                    if descriptor:
                        static_mesh = descriptor.get_editor_property("static_mesh")
                        if static_mesh:
                            static_mesh_name = clean_name(static_mesh.get_name())
                                
                            # Find the category for this mesh in TYPE_MAP
                            for type_name, meshes in parser_instance.TYPE_MAP.items():
                                if static_mesh_name in meshes:
                                    if type_name not in graph_data["types"]:
                                        graph_data["types"][type_name] = {"Meshes": meshes, "Nodes": []}

                                    for g_n in grouped_nodes:
                                        grouped_node = next((n for n in nodes if n.get_name() == g_n), None)
                                        graph_data["types"][type_name]["Nodes"].extend(
                                            [extract_node_data(grouped_node)]
                                        )
                                    break
                        else:
                            print(f"Warning: No static mesh found for spawner '{spawner_name}'.")
        else:
            print(f"Warning: '{spawner_name}' does not use a weighted mesh selector.")

    # for node in nodes:
    #     if node.get_name() not in grouped_nodes_instance:
    #         node_data = extract_node_data(node)
    #         if node_data:
    #             graph_data["types"].setdefault("Uncategorized", {"Meshes": [], "Nodes": []})["Nodes"].append(node_data)

    return graph_data

def save_all_graph_data_to_json(all_graph_data, save_directory="C:\\Users\\auror\\OneDrive\\Documents\\Unreal_Projects\\Gen_AI_Experiment\\Content\\Scripts", json_filename="all_graphs_data.json"):
    try:
        # Use the specified directory, or default to the current script's directory
        if save_directory is None:
            save_directory = os.path.dirname(__file__)

        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Construct the full file path
        save_path = os.path.join(save_directory, json_filename)

        # Write the JSON data to the file
        with open(save_path, 'w') as json_file:
            json.dump(all_graph_data, json_file, indent=4)

        unreal.log(f"Graph data successfully saved to: {save_path}")
    except Exception as e:
        unreal.log_error(f"Error saving graph data to {json_filename}: {str(e)}")

def get_undesirable_graphs():
    """
    Filters and retrieves PCG graphs with 'UNDESIRABLE_' in their names from the Unreal asset registry.
    """
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    # Retrieve all assets in the project under the "/Game" path
    all_assets = asset_registry.get_assets_by_path("/Game", recursive=True)

    # Regular expression to match asset names starting with 'UNDESIRABLE_'
    undesirable_pattern = re.compile(r"^UNDESIRABLE_")

    # Filter only PCGGraph assets that match the undesirable pattern
    undesirable_pcg_assets = [
        asset for asset in all_assets
        if "PCGGraph" in str(asset.asset_class_path) and undesirable_pattern.match(str(asset.asset_name))
    ]

    undesirable_pcg_graphs = []
    for asset in undesirable_pcg_assets:
        try:
            # Load the asset using get_asset()
            pcg_graph = asset.get_asset()
            if pcg_graph:
                undesirable_pcg_graphs.append(pcg_graph)
        except Exception as e:
            unreal.log_error(f"Error loading asset {asset.object_path}: {str(e)}")

    return undesirable_pcg_graphs

def save_filtered_graph_data(filtered_graph_data, save_directory="C:\\Users\\auror\\OneDrive\\Documents\\Unreal_Projects\\Gen_AI_Experiment\\Content\\Scripts", json_filename="undesirable_graphs_data.json"):
    """
    Saves only the filtered undesirable graph data to a separate JSON file.
    """
    try:
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Construct the full file path
        save_path = os.path.join(save_directory, json_filename)

        # Write the JSON data to the file
        with open(save_path, 'w') as json_file:
            json.dump(filtered_graph_data, json_file, indent=4)

        unreal.log(f"Filtered graph data successfully saved to: {save_path}")
    except Exception as e:
        unreal.log_error(f"Error saving filtered graph data to {json_filename}: {str(e)}")

# Parse each PCG graph and iterate through nodes.
def main():
    pcg_graphs = get_all_pcg_graphs()
    # undesirable_pcg_graphs = get_undesirable_graphs()

    # Set array for all data.
    all_graph_data = {"General": {}, "Undesirable": {}}
    # undesirable_graph_data = {}

    if not pcg_graphs:
        unreal.log_warning("No PCG Graphs found in the project.")
        return
    
    for pcg_graph in pcg_graphs:
        graph_name = pcg_graph.get_name()
        unreal.log(f"Processing PCG Graph: {graph_name}")
        try:
            if pcg_graph:
                parser_instance = MeshNameParser()
                node_tag_instance = TagSpawnerNode()

                # Extract graph data
                graph_data = extract_pcg_graph_data(pcg_graph, parser_instance, node_tag_instance)
                
                # Categorize the graph data
                if graph_name.startswith("UNDESIRABLE_"):
                    all_graph_data["Undesirable"][graph_name] = graph_data
                else:
                    all_graph_data["General"][graph_name] = graph_data

        except Exception as e:
            unreal.log_error(f"Error processing graph {graph_name}: {str(e)}")
            traceback.print_exc()
    
    save_all_graph_data_to_json(all_graph_data)

main()