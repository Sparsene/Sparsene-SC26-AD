import csv
import json
import argparse

parser = argparse.ArgumentParser(description="  ACC CSV     JSON")
parser.add_argument(
    "csv_path",
    nargs="?",
    default="/workspace/results/result.csv",
    help="   CSV     ",
)
parser.add_argument(
    "json_path",
    nargs="?",
    default="/workspace/results/acc_fp32.json",
    help="   JSON     ",
)
args = parser.parse_args()

csv_path = args.csv_path
json_path = args.json_path

method="acc"

results = {}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 3:
            continue
        filename = row[0]
        N = row[1]
        acc_value = float(row[2])

        #        
        if filename not in results:
            results[filename] = {}
        if N not in results[filename]:
            results[filename][N] = {method: []}

        #    
        results[filename][N][method].append(acc_value)

#    JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"    ，      {json_path}")
