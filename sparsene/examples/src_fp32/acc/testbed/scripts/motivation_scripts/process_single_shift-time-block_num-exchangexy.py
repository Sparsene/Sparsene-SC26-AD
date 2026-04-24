#!/usr/bin/env python3
import json
import re

#    JSON   
input_json = "plan_results.json"
#    exchange_xy   log   
log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/test_single_shift_exchangexy.log"
#    JSON   
output_json = "plan_results_exchange.json"

# 1️⃣      JSON
with open(input_json, "r", encoding="utf-8") as f:
    plan_results = json.load(f)

# 2️⃣   exchange_xy log     
exchange_data = {}
current_plan = None
current_time = None

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        #    plan   
        match_plan = re.search(r"plan_(\d+)", line)
        if match_plan:
            current_plan = int(match_plan.group(1))
            current_time = None
            continue

        #    mykernel_time
        match_time = re.search(r"mykernel_time:\s*([\d.]+)\s*ms", line)
        if match_time and current_plan is not None:
            current_time = float(match_time.group(1))
            continue

        #    active block num
        match_block = re.search(r"active block num:\s*(\d+)", line)
        if match_block and current_plan is not None and current_time is not None:
            block_num = int(match_block.group(1))
            #     ，      
            exchange_data[current_plan] = {
                "time_ms_exchange_xy": current_time,
                "block_num_exchange_xy": block_num
            }
            current_plan = None
            current_time = None

# 3️⃣        JSON
for item in plan_results:
    plan_id = item["plan_id"]
    if plan_id in exchange_data:
        item.update(exchange_data[plan_id])
    else:
        #         exchange_xy   ，    None
        item["time_ms_exchange_xy"] = None
        item["block_num_exchange_xy"] = None

# 4️⃣       JSON   
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(plan_results, f, indent=2)

print(f"       {output_json}")
