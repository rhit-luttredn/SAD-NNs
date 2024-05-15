#!/usr/bin/env python3
import subprocess
import sys

def main(script_name, vectorized_args):
    # Split the vectorized_args list by '--' to separate argument sets
    args_sets = []
    current_args = []
    for arg in vectorized_args:
        if arg == "--":
            if current_args:  # if current_args is not empty
                args_sets.append(current_args)
                current_args = []
        else:
            current_args.append(arg)
    if current_args:  # Add the last set of arguments if any
        args_sets.append(current_args)

    # Iterate through each set of arguments in the list
    for args in args_sets:
        # Prepare the command to run the script with current arguments
        command = ['python3', script_name] + args
        print(f"Running: {' '.join(command)}")
        
        # Execute the command
        subprocess.run(command)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run-vectorized.py <script_name.py> <args_for_script> [-- <args_for_script> ...]")
        sys.exit(1)
    
    script_name = sys.argv[1]
    vectorized_args = sys.argv[2:]
    
    main(script_name, vectorized_args)
