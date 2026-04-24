import csv
import json
import argparse
parser = argparse.ArgumentParser(description="  DTC CSV     JSON")
parser.add_argument(
    "csv_path",
    nargs="?",
    default="/workspace/results/result.csv",
    help="   CSV     ",
)
parser.add_argument(
    "json_path",
    nargs="?",
    default="/workspace/results/dtc_fp32.json",
    help="   JSON     ",
)
args = parser.parse_args()

csv_path = args.csv_path
json_path = args.json_path

results = {}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 4:
            continue
        filename = row[0]
        N = row[1]
        value = float(row[3])

        #        
        if filename not in results:
            results[filename] = {}
        if N not in results[filename]:
            results[filename][N] = {"dtc": []}

        #    
        results[filename][N]["dtc"].append(value)

#    JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"    ，      {json_path}")
