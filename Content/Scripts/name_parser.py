class MeshNameParser:
    def __init__(self):
        # Define a type map for static meshes.
        self.TYPE_MAP = {
            "Grass": [],
            "GroundItems": [],
            "Structures": [],
            "Trees": []
        }

        # Keyword-to-type mapping.
        self.KEYWORD_TO_TYPE = {
            "Grass": ["grass"],  # Ensure keywords like "grass" are defined
            "GroundItems": ["rock", "stone", "bush"],
            "Trees": ["tree"],
            "Structures": ["temple", "relief", "sculpture", "column", "pottery"],
        }

    def associate_name_with_keyword(self, name):
        try:
            if not isinstance(name, str):
                raise TypeError(f"Expected 'name' to be a string, but got {type(name)}.")
            name = name.strip()
            name_lower = name.lower()
            added = False
            for category, keywords in self.KEYWORD_TO_TYPE.items():
                print(f"Checking category '{category}' with keywords: {keywords} for name: '{name_lower}'")
                if any(keyword in name_lower for keyword in keywords):
                    # Ensure the category exists in TYPE_MAP
                    if category.lower() not in map(str.lower, self.TYPE_MAP.keys()):
                        raise KeyError(f"Category '{category}' not found in TYPE_MAP.")
                    
                    self.TYPE_MAP[category].append(name)
                    print(f"Added '{name}' to '{category}' based on keywords {keywords}.")
                    added = True
                    break
            if not added:
                print(f"No matching keyword found for '{name}'. Skipped.")
        except Exception as e:
            print(f"Error occurred: {e}")

    def print_type_map(self):
        print("\n--- Current TYPE_MAP ---")
        for category, names in self.TYPE_MAP.items():
            print(f"{category}: {names if names else 'No entries'}")
        print("-------------------------")