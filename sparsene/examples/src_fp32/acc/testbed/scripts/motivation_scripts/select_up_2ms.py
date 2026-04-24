import re

log_file = "/workspace/sparsene/examples/src_fp32/acc/testbed/build/cumemcheck.1.log"   #            

plan_times = []
current_plan = None

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        #    plan   
        match_plan = re.search(r"acc-plan_(\d+)-tf32", line)
        if match_plan:
            current_plan = int(match_plan.group(1))
            continue
        
        #    mykernel_time
        match_time = re.search(r"mykernel_time:\s*([\d.]+)\s*ms", line)
        if match_time and current_plan is not None:
            time_ms = float(match_time.group(1))
            # if time_ms < 1.35:
            plan_times.append((current_plan, time_ms))
            current_plan = None   #   plan      ，  
    
#     
# print("mykernel_time > 2 ms   plan   ：")
# for plan, t in plan_times:
#     print(f"plan_{plan}: {t:.6f} ms")

plan_times.sort(key=lambda x: x[1])
bucket_plan_times = {}
bucket_size = 0.1
for time_end in range(11, 50):
    bucket_plan_times[time_end] = []

for plan, t in plan_times:
    if plan == 0:
        print(t, int(t * 10))
    bucket_num = int(t * 10)
    bucket_plan_times[bucket_num].append((plan, t))

for time_end in range(11, 23):
    # if len(bucket_plan_times[time_end]) == 0: 
    #     continue
    f = open("/workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/plan_fig/" + str(time_end // 10) + "_" + str(time_end % 10) + "ms/plans.txt", "w")
    for plan, t in bucket_plan_times[time_end]:
        f.write(f"{plan}\n")

# for plan, t in plan_times:
#     print(f"{plan}")