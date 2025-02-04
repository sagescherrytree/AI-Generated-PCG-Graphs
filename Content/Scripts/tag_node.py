import unreal
import traceback

# Import all tag logic to this script.

class TagSpawnerNode:

    def fill_spawner_nodes(self, pcg_graph, spawner_positions, grouped_nodes):
        try:
            for node in pcg_graph.nodes:
                settings = node.get_settings()
                if isinstance(settings, unreal.PCGStaticMeshSpawnerSettings):
                    spawner_name = node.get_name()
                    print(f"StaticMeshSpawner found: {spawner_name}")
                    spawner_positions[spawner_name] = node.get_node_position()
                    grouped_nodes[spawner_name] = [] # Initialize group for this sampler
                else:
                    print("No specific settings for this node.")
        except Exception as e:
            print(f"Error processing Surface Sampler node: {e}")
            traceback.print_exec()

    def check_node_proximity_to_static_mesh_spawner(self, node, spawner_positions, y_threshold=20):
        try:
            node_x, node_y = node.get_node_position()
            close_nodes_to_spawner = []

            for spawner_name, (_, sampler_y) in spawner_positions.items():
                if abs(node_y - sampler_y) <= y_threshold:
                    close_nodes_to_spawner.append(spawner_name)
                    print(f"Appended node {node.get_name()} to close_nodes_to_spawner for {spawner_name}.")

            return close_nodes_to_spawner
        except Exception as e:
            print(f"Error checking proximinity for node '{node.get_name()}': {e}")
            traceback.print_exce()
            return []