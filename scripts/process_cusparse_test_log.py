import re
import json
import sys
from pathlib import Path

def parse_results(logfile, dtype="fp32"):
    """
          ，   matrix, N, executable, mykernel_time
    """

    results = {}

    with open(logfile, "r", encoding="utf-8") as f:
        content = f.read()

    #        
    sections = re.split(r"=+\s*Matrix:", content)
    for sec in sections:
        if not sec.strip():
            continue
        sec = "Matrix:" + sec  #      

        #        (timed out)
        if "timed out after" in sec:
            continue

        #    matrix filename
        m_matrix = re.search(r"Matrix:\s*(\S+)", sec)
        m_exec   = re.search(r"Executable:\s*(\S+)", sec)
        m_n      = re.search(r"N:\s*(\d+)", sec)
        m_time   = re.search(r"mykernel_time:\s*([0-9.]+)", sec)

        if not (m_matrix and m_exec and m_n and m_time):
            continue

        matrix = Path(m_matrix.group(1)).name  #       
        exe    = Path(m_exec.group(1)).name
        n      = m_n.group(1)
        time   = float(m_time.group(1))

        #         (CUDA API failed / Error(...))
        if "CUDA API failed" in sec or "Error(" in sec:
            continue

        #     
        results.setdefault(matrix, {}).setdefault(n, {})["cusp"] = time

    return results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("  : python parse_log.py <logfile> <output_json_file> fp32/fp16")
        sys.exit(1)

    logfile = sys.argv[1]
    json_dir = sys.argv[2]
    dtype = sys.argv[3]
    

    out_json = json_dir

    results = parse_results(logfile, dtype)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"    ，       {out_json}")
