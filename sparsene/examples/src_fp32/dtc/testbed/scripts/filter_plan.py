import re

#       ，    plan
input_file = "/workspace/sparsene/examples/src_fp32/dtc/testbed/plans.txt"
output_file = "filtered_plans.txt"

pattern_num = re.compile(r"^(\d+),")        #    plan   
pattern_x   = re.compile(r"\|\((\d+)\)>")   #      x

valid_plans = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        #     
        m = pattern_num.match(line)
        if not m:
            continue
        plan_id = m.group(1)

        #      (x)
        numbers = pattern_x.findall(line)

        #           "1"
        if numbers and all(n == "1" for n in numbers):
            valid_plans.append((plan_id, line))

#    CSV，      ，    plan  
with open(output_file, "w", encoding="utf-8") as f:
    # f.write("plan_id,plan\n")
    for plan_id, plan in valid_plans:
        print(f"\"${{PROJECT_SOURCE_DIR}}/plans/plan_{plan_id}.inc\"")
        f.write(f"{plan}\n")


print(f"    ，    {len(valid_plans)}        plan，     {output_file}")
