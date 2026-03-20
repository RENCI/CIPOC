import glob
import json

if __name__ == "__main__":
    json_files = glob.glob("coding_rules/*.json")
    for file in json_files:
        with open(file, "r") as in_file:
            content = json.load(in_file)
        
        if isinstance(content, str):
            content = json.loads(content)

            with open(file, "w") as out_file:
                json.dump(content, out_file, indent=2)