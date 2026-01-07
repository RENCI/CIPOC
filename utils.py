import os
import json
import glob
from collections import defaultdict
from typing import Literal


def find_output_files(output_type: Literal["results", "error"], directory: str = ".") -> dict | None:
    output_files = glob.glob(f"*{output_type}*.json")
    if not output_files:
        return None
    
    output_file_dict = defaultdict(list)
    for file in output_files:
        model_name = file.split("_")[0]
        output_file_dict[model_name].append(file)

    return dict(output_file_dict)

def get_model_run_state(model_name: str, state_type: Literal["results", "error"] = "results", output_directory: str = ".") -> dict | None:
    """ Get number of results for each patient for a given model """
    results = find_output_files(state_type, output_directory)
    if results is None:
        return None
    
    model_state = defaultdict(int)
    for file in results[model_name]:
        with open(file, "r") as f:
            for line in f.readlines():
                line_json = json.loads(line)
                if state_type == "results" or (state_type == "error" and line_json["explanation"] == "Error"):
                    model_state[line_json["patient_id"]] += 1

    return dict(model_state)
