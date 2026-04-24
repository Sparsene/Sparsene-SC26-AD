import csv
import json
import sys

def csv_to_json(csv_file, json_file, N=128):
    data = {}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_name = row['dataSet'].strip()
            sputnik_time = float(row['sputnik'])
            data[dataset_name] = {
                str(N): {
                    "sputnik": sputnik_time
                }
            }

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("  : python script.py input.csv output.json N")
    else:
        csv_to_json(sys.argv[1], sys.argv[2], sys.argv[3])
