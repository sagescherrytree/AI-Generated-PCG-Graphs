import unreal
import os
import sys

# Specify the path we want to set as current work directory.
new_working_directory = r"C:\Users\auror\OneDrive\Documents\Unreal_Projects\Gen_AI_Experiment\Content\Scripts"

# Change the working directory.
os.chdir(new_working_directory)

# Confirm change.
print("Current working directory: ", os.getcwd())

# Get the directory of current script.
script_directory = r"C:\Users\auror\OneDrive\Documents\Unreal_Projects\Gen_AI_Experiment\Content\Scripts"

# Add directory to sys.path if not already present.
if script_directory not in sys.path:
    sys.path.append(script_directory)

print("sys.path: ", sys.path)
print("Current working directory: ", os.getcwd())