import csv
import json
import argparse

def csv_to_json(csv_path, json_path, N):
    results = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["dataSet"]
            
            #         （     ）
            times = [float(row[col]) for col in row if col not in ("dataSet", "num_nodes", "num_edges")]
            
            if filename not in results:
                results[filename] = {}
            # N    CSV  ，        
            results[filename][N] = {"flashsparse": times}

    #    json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to JSON with fixed N")
    parser.add_argument("csv_path", help="   CSV     ")
    parser.add_argument("json_path", help="   JSON     ")
    parser.add_argument("-N", type=int, required=True, help="   N   ")
    args = parser.parse_args()

    csv_to_json(args.csv_path, args.json_path, args.N)
    print(f"    ，      {args.json_path},    N={args.N}")
