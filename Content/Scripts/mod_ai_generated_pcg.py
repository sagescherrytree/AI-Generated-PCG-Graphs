import unreal
import json
from collections import defaultdict

# Define a dictionary (hashmap) where keys are node types and values are lists of relevant parameters
PARAMETER_MAP = {
    "PCGSurfaceSamplerSettings": ["point_extents", "points_per_squared_meter"],
    "PCGSpatialNoiseSettings": ["random_offset", "transform"],
    "PCGDensityFilterSettings": ["lower_bound", "upper_bound", "invert_filter"],
    "PCGTransformPointsSettings": ["offset_min", "offset_max", "rotation_min", "rotation_max", "scale_min", "scale_max"]
}

# TEMPORARY
# -------------------
# Define a dictionary mapping nodes from graph to type.
#Only contains nodes that we need to modify (see dict above w/ associated params)

GRAPH_TO_MOD_NODE_TO_TYPE = {
    "SurfaceSampler_0": "Grass",
    "Spatial Noise_1": "Grass", 
    "TransformPoints_3": "Grass",
    "SurfaceSampler_10": "GroundItems",
    "Spatial Noise_11": "GroundItems",
    "DensityFilter_7": "GroundItems",
    "TransformPoints_8": "GroundItems",
    "SurfaceSampler_13": "Structures",
    "DensityFilter_14": "Structures",
    "TransformPoints_16": "Structures",
    "SurfaceSampler_21": "Trees",
    "DensityFilter_25": "Trees",
    "TransformPoints_22": "Trees"
}

actor_name = "BP_Test_AI_Gen_Actor_PCG"

bp_actor = None
pcg_component = None
pcg_graph = None

# Get actor subsystem:
actor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# Get selected actor
actors_list = actor_subsys.get_all_level_actors()

# Iterate through actors to find target actor that matches w/ actor_name.
for actor in actors_list:
    actorLabel = actor.get_actor_label()
    if actorLabel == actor_name:
        bp_actor = actor

if isinstance(bp_actor, unreal.Actor):
    print("Target actor found: ", bp_actor)
    # Get the component list of the actor
    component_list = bp_actor.get_components_by_class(unreal.PCGComponent)

    # Print out the components
    print("Components:")
    for component in component_list:
        print(component.get_class().get_name())
    
    if component_list.__sizeof__ != 0:
        # Can get the first component found b/c only one PCG component in this actor.
        pcg_component = component_list[0]
        print("PCG component found: ", pcg_component.get_name())

    # Get the PCG Graph referenced by the PCG Component
    pcg_graph = pcg_component.graph_instance.graph.get_const_pcg_graph()
    if pcg_graph:
        print("PCG Graph found: ", pcg_graph.get_name())
        print("Type: ", type(pcg_graph).__name__)
    else:
        print("No PCG Graph found for the selected PCG Component")
else:
    print("Selected asset is not an Actor, it is a: ", type(bp_actor).__name__)

def find_node_by_name(pcg_graph, node_name):
    for node in pcg_graph.nodes:  # Assuming get_nodes() returns a list of all nodes
        if node.get_name() == node_name:  # Replace with the correct property/method for the node's name
            return node
    return None

# Point extents and points per squared metre come from regenerated graph data.
def modify_surface_sampler(node_settings, param, value):
    if not node_settings:
        print("No pcg graph node found.")
        return
    
    try:

        point_extents = None
        points_per_squared_meter = None
        
        # Update point_extents
        if param == "point_extents":
            point_extents = value
            current_value = node_settings.get_editor_property("point_extents")
            extents_vector = unreal.Vector(point_extents["x"], point_extents["y"], point_extents["z"])
            print(f"  point_extents: Current Value = {current_value}, New Value = {extents_vector}")
            node_settings.set_editor_property("point_extents", extents_vector)

        # Update points_per_squared_meter
        if param == "points_per_squared_meter":
            points_per_squared_meter = value
            current_value = node_settings.get_editor_property("points_per_squared_meter")
            print(f"  points_per_squared_meter: Current Value = {current_value}, New Value = {float(points_per_squared_meter)}")
            node_settings.set_editor_property("points_per_squared_meter", float(points_per_squared_meter))

    except Exception as e:
        print(f"  Failed to update properties for PCGSurfaceSamplerSettings: {e}")

# Modify spatial noise node.
def modify_spatial_noise(node_settings, param, value):
    if not node_settings:
        print("No pcg graph node found.")
        return 
    
    try:
        random_offset = None                           
        transform = None

        # Update random_offset
        if param == "random_offset":
            random_offset = value
            rand_offset_vec = unreal.Vector(random_offset["x"], random_offset["y"], random_offset["z"])
            current_value = node_settings.get_editor_property("random_offset")
            print(f"  random_offset: Current Value = {current_value}, New Value = {rand_offset_vec}")
            node_settings.set_editor_property("random_offset", rand_offset_vec)

        # Update transform.
        if param == "transform":
            transform = value

            translation = transform.get("translation", [0, 0, 0])
            rotation = transform.get("rotation", [0, 0, 0])
            scale = transform.get("scale", [1, 1, 1])

            translation_vec = unreal.Vector(*translation) if len(translation) == 3 else unreal.Vector(0, 0, 0)
            rotation_rotator = unreal.Rotator(*rotation) if len(rotation) == 3 else unreal.Rotator(0, 0, 0)
            scale_vec = unreal.Vector(*scale) if len(scale) == 3 else unreal.Vector(1, 1, 1)

            new_transform = unreal.Transform(translation_vec, rotation_rotator, scale_vec)

            current_value = node_settings.get_editor_property("transform")
            print(f"  transform: Current Value = {current_value}, New Value = {new_transform}")
            node_settings.set_editor_property("transform", new_transform)
    except Exception as e:
        print(f"  Failed to update properties for PCGSpatialNoiseSettings: {e}")

# Modify density filter node.
def modify_density_filter(node_settings, param, value):
    if not node_settings:
        print("No pcg graph node found.")
        return 
    
    try:
        lower_bound = None
        upper_bound = None
        invert_filter = None

        # Update lower_bound
        if param == "lower_bound":
            lower_bound = value
            current_value = node_settings.get_editor_property("lower_bound")
            print(f"  lower_bound: Current Value = {current_value}, New Value = {float(lower_bound)}")
            node_settings.set_editor_property("lower_bound", float(lower_bound))

        # Update upper_bound.
        if param == "upper_bound":
            upper_bound = value
            current_value = node_settings.get_editor_property("upper_bound")
            print(f"  upper_bound: Current Value = {current_value}, New Value = {float(upper_bound)}")
            node_settings.set_editor_property("upper_bound", float(upper_bound))

        # Update invert_filter.
        if param == "invert_filter":
            invert_filter = value
            current_value = node_settings.get_editor_property("invert_filter")
            print(f"  invert_filter: Current Value = {current_value}, New Value = {float(invert_filter)}")
            node_settings.set_editor_property("invert_filter", float(invert_filter))
    except Exception as e:
        print(f"  Failed to update properties for PCGSpatialNoiseSettings: {e}")

# Modify transform points node.
def modify_transform_points(node_settings, param, value):
    if not node_settings:
        print("No pcg graph node found.")
        return 
    
    try:
        offset_min = None
        offset_max = None
        rotation_min = None
        rotation_max = None
        scale_min = None
        scale_max = None

        # Update offset_min
        if param == "offset_min":
            offset_min = value
            offset_min_vector = unreal.Vector(offset_min["x"], offset_min["y"], offset_min["z"])
            current_value = node_settings.get_editor_property("offset_min")
            print(f"  offset_min: Current Value = {current_value}, New Value = {offset_min_vector}")
            node_settings.set_editor_property("offset_min", offset_min_vector)

        # Update offset_max.
        if param == "offset_max":
            offset_max = value
            offset_max_vector = unreal.Vector(offset_max["x"], offset_max["y"], offset_max["z"])
            current_value = unreal.Vector(offset_max["x"], offset_max["y"], offset_max["z"])
            print(f"  offset_max: Current Value = {current_value}, New Value = {offset_max_vector}")
            node_settings.set_editor_property("offset_max", offset_max_vector)

        # Update rotation_min.
        if param == "rotation_min":
            rotation_min = value
            current_value = node_settings.get_editor_property("rotation_min")
            print(f"  rotation_min: Current Value = {current_value}, New Value = {rotation_min}")
            node_settings.set_editor_property("rotation_min", rotation_min)

        # Update rotation_max.
        if param == "rotation_max":
            rotation_max = value
            current_value = node_settings.get_editor_property("rotation_max")
            print(f"  rotation_max: Current Value = {current_value}, New Value = {rotation_max}")
            node_settings.set_editor_property("rotation_max", rotation_max)

        # Update scale_min.
        if param == "scale_min":
            scale_min = value
            scale_min_vector = unreal.Vector(scale_min["x"], scale_min["y"], scale_min["z"])
            current_value = node_settings.get_editor_property("scale_min")
            print(f"  scale_min: Current Value = {current_value}, New Value = {scale_min_vector}")
            node_settings.set_editor_property("scale_min", scale_min_vector)

        # Update scale_max.
        if param == "scale_max":
            scale_max = value
            scale_max_vector = unreal.Vector(scale_max["x"], scale_max["y"], scale_max["z"])
            current_value = node_settings.get_editor_property("scale_max")
            print(f"  scale_max: Current Value = {current_value}, New Value = {scale_max_vector}")
            node_settings.set_editor_property("scale_max", scale_max_vector)
    except Exception as e:
        print(f"  Failed to update properties for PCGTransformPointsSettings: {e}")

# Main function for calling mod items.
if pcg_graph:
    print("We have a PCG Graph! ", pcg_graph.get_name())

json_filename = "reconstructed_graph.json"
with open(json_filename, "r") as f:
    json_data = json.load(f)

def update_pcg_graph(pcg_graph, json_data, node_to_type_mapping):
    type_to_nodes = defaultdict(list)
    for node_name, node_type in node_to_type_mapping.items():
        type_to_nodes[node_type].append(node_name)

    # Iterate through the JSON types
    for type_name, type_data in json_data["types"].items():
        print(f"Processing Type: {type_name}")
        
        # Check if there are nodes for this type in the mapping
        if type_name not in type_to_nodes:
            print(f"  No nodes in the mapping for type: {type_name}")
            continue
        
        # Nodes to be updated for this type
        nodes_to_update = type_to_nodes[type_name]
        
        # Iterate through the JSON nodes of this type
        for node_data in type_data["Nodes"]:
            node_settings_type = node_data["node_settings"]
            parameters = node_data["parameters"]
            for node_name in nodes_to_update:
                if node_name in node_to_type_mapping and node_to_type_mapping[node_name] == type_name:

                    node = find_node_by_name(pcg_graph, node_name)
                    if not node:
                        print(f"  Node {node_name} not found in PCG graph. Skipping.")
                        continue

                    node_settings = node.get_settings()
                    if not node_settings:
                        print(f"  Node settings not found for {node_name}. Skipping.")
                        continue

                    print(f"  Node: {node_name} | Expected Settings Type: {node_settings_type} | Actual Settings: {type(node_settings).__name__}")
                    
                    for param, value in parameters.items():
                        print(f"    Param: {param}, Value: {value}")
                            
                        if isinstance(node_settings, unreal.PCGSurfaceSamplerSettings) and node_settings_type == "PCGSurfaceSamplerSettings":
                            # modify_surface_sampler(node_settings, point_extents, points_per_squared_meter)
                            modify_surface_sampler(node_settings, param, value)
                        elif isinstance(node_settings, unreal.PCGSpatialNoiseSettings) and node_settings_type == "PCGSpatialNoiseSettings":
                            # modify_spatial_noise(node_settings, random_offset, transform)
                            modify_spatial_noise(node_settings, param, value)
                        elif isinstance(node_settings, unreal.PCGDensityFilterSettings) and node_settings_type == "PCGDensityFilterSettings":
                            # modify_density_filter(node_settings, lower_bound, upper_bound, invert_filter)
                            modify_density_filter(node_settings, param, value)

                        elif isinstance(node_settings, unreal.PCGTransformPointsSettings) and node_settings_type == "PCGTransformPointsSettings":
                            # modify_transform_points(node_settings, offset_min, offset_max, rotation_min, rotation_max, scale_min, scale_max )
                            modify_transform_points(node_settings, param, value)

if pcg_graph:
    update_pcg_graph(pcg_graph, json_data, GRAPH_TO_MOD_NODE_TO_TYPE)
    try:
        pcg_component.generate(True)
        print("PCG graph successfully regenerated.")
    except Exception as e:
        print(f"Failed to regenerate the PCG graph: {e}")

else:
    print("No PCG graph found.")