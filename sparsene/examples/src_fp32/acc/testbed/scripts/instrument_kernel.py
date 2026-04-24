#!/usr/bin/env python3
"""
instrument_kernel.py — Add clock() timestamps around each op call in kernel.inc
================================================================================

Reads kernel.inc, wraps each op call (marked by ``// op_call`` comments) with
INSTR_BEGIN() / INSTR_END() macros, and threads ``int& _instr_idx`` (register
counter) and ``InstrTrace* _instr_buf`` (static shared memory buffer) through
the entire pipeline call chain.

Additionally writes a metadata JSON file (<output>_meta.json) that records
the sequential order of instrumented ops.

Usage:
    python instrument_kernel.py kernel.inc kernel_instrumented.inc
"""

import json
import re
import sys

# Op name → op_id mapping (must match perf_model.py)
OP_IDS = {
    "G2sSparseIndexLoadOp": 0,
    "G2rSparseMcoOffLoadOp": 1,
    "G2rSparseMcoMaskLoadOp": 2,
    "G2sSparseMcoValLoadOp": 3,
    "G2sMatrixBLoadOp": 4,
    "S2sRestoreMatrixAOp": 5,
    "S2rAValLoadOp": 6,
    "S2rBValLoadOp": 7,
    "CalculateOp": 8,
}

# Regex to match op_call comments
OP_CALL_RE = re.compile(
    r"^(\s*)// op_call \|buf_idx=\d+\| (.+?) (\w+)\s*$"
)

# Pipeline functions within AccMainLoopOp that need instr params
PIPELINE_FN_RE = re.compile(
    r"^(\s+)CUTE_DEVICE void "
    r"(fill|loop_step|short_pipe|dispatch|remainder"
    r"|remains_\d+|empty_after_remain_\d+|empty)"
    r"\((.*?)\)\s*\{"
)

# AccMainLoopOp::f(int l, int r)
ACC_F_DEF_RE = re.compile(
    r"^(\s+)CUTE_DEVICE void f\(int l, int r\)\s*\{"
)

# blk_x_loop / blk_y_loop body() and f()
LOOP_BODY_DEF_RE = re.compile(
    r"^(\s+)CUTE_DEVICE void body\(\)\s*\{"
)
LOOP_F_DEF_RE = re.compile(
    r"^(\s+)CUTE_DEVICE void f\(\)\s*\{"
)

# Simple pipeline function CALLS within dispatch()
PIPELINE_CALL_RE = re.compile(
    r"^(\s+)(fill|loop_step|short_pipe|remainder"
    r"|remains_\d+|empty_after_remain_\d+|empty)"
    r"\((.+?)\);\s*$"
)

# dispatch() call within AccMainLoopOp::f
DISPATCH_CALL_RE = re.compile(r"^(\s+)dispatch\(\);\s*$")

# Template f<>() calls for AccMainLoopOp_v, blk_x_loop_v, blk_y_loop_v
TEMPLATE_CALL_RE = re.compile(
    r"^(\s+)(AccMainLoopOp_v|acc_spmm_kernel_blk_x_loop_v"
    r"|acc_spmm_kernel_blk_y_loop_v)\.template f<>\("
)

# __global__ function last param line (before closing paren)
GLOBAL_LAST_PARAM_RE = re.compile(
    r"^(\s+)(int K.*nnz_block)\s*$"
)

INSTR_PARAMS = "int& _instr_idx, InstrTrace* _instr_buf"
INSTR_ARGS = "_instr_idx, _instr_buf"


def instrument(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        lines = f.readlines()

    output: list[str] = []
    meta_sequence: list[dict] = []
    i = 0
    header_inserted = False
    in_global_fn = False
    global_shared_inserted = False
    n_instrumented = 0
    current_fn = ""
    global_fn_indent = "    "
    # Track which class we're in for body()/f() disambiguation
    in_blk_loop = False  # True when inside blk_x_loop or blk_y_loop

    while i < len(lines):
        line = lines[i]

        # ---- Insert #include after "using namespace cute;" ----
        if not header_inserted and "using namespace cute;" in line:
            output.append(line)
            output.append("\n")
            output.append("// ---- Instrumentation (auto-generated) ----\n")
            output.append('#include "instrumentation.cuh"\n')
            output.append("\n")
            header_inserted = True
            i += 1
            continue

        # ---- Track class context ----
        if "class acc_spmm_kernel_blk_x_loop" in line or \
           "class acc_spmm_kernel_blk_y_loop" in line:
            in_blk_loop = True
        elif line.startswith("class ") or (re.match(r"^template\s*<", line) and
              i + 1 < len(lines) and "class " in lines[i + 1]):
            # Entering a non-loop class
            pass  # in_blk_loop stays until we see the next class
        if re.match(r"^class\s+\w+", line) and "blk_x_loop" not in line \
                and "blk_y_loop" not in line:
            in_blk_loop = False

        # ---- Track __global__ function ----
        if "__global__" in line:
            in_global_fn = True

        # ---- Add kernel params to __global__ function ----
        # Match the last parameter line: "    int K, int M, int Mo, int N, int nnz, int nnz_block"
        if in_global_fn and not global_shared_inserted:
            m = GLOBAL_LAST_PARAM_RE.match(line)
            if m:
                # Add instr params
                indent = m.group(1)
                output.append(line.rstrip() + ",\n")
                output.append(f"{indent}InstrTrace* _d_instr_trace, int* _d_instr_count, int _instr_nblocks\n")
                i += 1
                continue

        # ---- Insert register counter after "const int wid" ----
        # NOTE: __shared__ InstrTrace _instr_buf is inserted AFTER all kernel
        # __shared__ declarations (second pass) so the trace buffer sits at the
        # END of shared memory, not the beginning.
        if in_global_fn and not global_shared_inserted and \
                "const int wid" in line:
            output.append(line)
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"\n{indent}int _instr_idx = 0;\n\n")
            global_shared_inserted = True
            global_fn_indent = indent
            i += 1
            continue

        # ---- Track current function name ----
        fn_match = re.match(r"\s+CUTE_DEVICE void (\w+)\(", line)
        if fn_match:
            current_fn = fn_match.group(1)

        # ---- Modify pipeline function DEFINITIONS ----
        m = PIPELINE_FN_RE.match(line)
        if m:
            indent = m.group(1)
            fname = m.group(2)
            params = m.group(3).strip()
            if params:
                new_line = f"{indent}CUTE_DEVICE void {fname}({params}, {INSTR_PARAMS}) {{\n"
            else:
                new_line = f"{indent}CUTE_DEVICE void {fname}({INSTR_PARAMS}) {{\n"
            output.append(new_line)
            i += 1
            continue

        # ---- Modify AccMainLoopOp::f(int l, int r) definition ----
        m = ACC_F_DEF_RE.match(line)
        if m and not in_blk_loop:
            indent = m.group(1)
            output.append(f"{indent}CUTE_DEVICE void f(int l, int r, {INSTR_PARAMS}) {{\n")
            i += 1
            continue

        # ---- Modify blk_loop body() and f() definitions ----
        m = LOOP_BODY_DEF_RE.match(line)
        if m and in_blk_loop:
            indent = m.group(1)
            output.append(f"{indent}CUTE_DEVICE void body({INSTR_PARAMS}) {{\n")
            i += 1
            continue

        m = LOOP_F_DEF_RE.match(line)
        if m and in_blk_loop:
            indent = m.group(1)
            output.append(f"{indent}CUTE_DEVICE void f({INSTR_PARAMS}) {{\n")
            i += 1
            continue

        # ---- Modify pipeline function CALLS ----
        m = PIPELINE_CALL_RE.match(line)
        if m:
            indent = m.group(1)
            fname = m.group(2)
            args = m.group(3)
            output.append(f"{indent}{fname}({args}, {INSTR_ARGS});\n")
            i += 1
            continue

        # ---- Modify dispatch() call ----
        m = DISPATCH_CALL_RE.match(line)
        if m:
            indent = m.group(1)
            output.append(f"{indent}dispatch({INSTR_ARGS});\n")
            i += 1
            continue

        # ---- Modify template f<>() calls ----
        m = TEMPLATE_CALL_RE.match(line)
        if m:
            indent = m.group(1)
            obj = m.group(2)
            # Collect all lines until ");", then insert instr args
            call_lines = [line]
            j = i + 1
            while j < len(lines) and not lines[j - 1].rstrip().endswith(");"):
                call_lines.append(lines[j])
                j += 1
            # call_lines now contains the full call.
            # Find the last ");", insert args before it.
            last = call_lines[-1]
            # Check if call is single-line: "obj.template f<>();"
            if last.rstrip().endswith(">();"):
                # No existing args, just add ours
                new_last = last.replace(">();", f">({INSTR_ARGS});")
                call_lines[-1] = new_last
            else:
                # Multi-line: last line ends with ");"
                # Insert args on a new line before ");"
                close_indent = last[: len(last) - len(last.lstrip())]
                # Remove the ");" from last line, add comma + args
                stripped = last.rstrip()
                if stripped.endswith(");"):
                    # Replace ");" with ",\n    ARGS\n);"
                    pre = stripped[:-2]  # everything before ");"
                    call_lines[-1] = pre + ",\n"
                    call_lines.append(f"{close_indent}    {INSTR_ARGS}\n")
                    call_lines.append(f"{close_indent});\n")

            for cl in call_lines:
                output.append(cl)
            i = j
            continue

        # ---- Modify blk_loop body() calls ----
        body_call_m = re.match(r"^(\s+)body\(\);\s*$", line)
        if body_call_m and in_blk_loop:
            indent = body_call_m.group(1)
            output.append(f"{indent}body({INSTR_ARGS});\n")
            i += 1
            continue

        # ---- Insert flush after top-level call in __global__ ----
        # The top-level call is "acc_spmm_kernel_blk_y_loop_v.template f<>(...)"
        # After modifying it above, we also need to insert flush code.
        # Detect: line after the closing "};" of blk_y_loop_v.template f<>() call
        # Actually, detect the pattern after the last template call modification.
        # Simpler: detect the closing of the __global__ function — "};" at col 0
        if in_global_fn and line.strip() == "};" and not line.startswith(" "):
            # Insert flush before closing
            # fp32 acc: grid is (threadblock_num_y, threadblock_num_x), so
            # block_linear_idx = blockIdx.y * gridDim.x + blockIdx.x
            output.append("    // ---- Flush instrumentation to GMEM (auto-generated) ----\n")
            output.append("    {\n")
            output.append("        int _blk_lin = blockIdx.y * gridDim.x + blockIdx.x;\n")
            output.append("        if (_blk_lin < _instr_nblocks && _d_instr_trace != nullptr) {\n")
            output.append("            InstrTrace* _dst = _d_instr_trace + _blk_lin * MAX_INSTR_ENTRIES;\n")
            output.append("            for (int _i = 0; _i < _instr_idx; _i++) {\n")
            output.append("                _dst[_i] = _instr_buf[_i];\n")
            output.append("            }\n")
            output.append("            _d_instr_count[_blk_lin] = _instr_idx;\n")
            output.append("        }\n")
            output.append("    }\n")
            output.append(line)
            in_global_fn = False
            i += 1
            continue

        # ---- Wrap op calls with INSTR_BEGIN / INSTR_END ----
        m = OP_CALL_RE.match(line)
        if m:
            indent = m.group(1)
            iter_expr = m.group(2).strip()
            op_name = m.group(3).strip()

            if op_name in OP_IDS:
                op_id = OP_IDS[op_name]

                meta_sequence.append({
                    "op_id": op_id,
                    "op_name": op_name,
                    "iteration": iter_expr,
                    "function": current_fn,
                })

                # Emit original comment
                output.append(line)
                i += 1

                # Open block + INSTR_BEGIN
                output.append(f"{indent}{{ INSTR_BEGIN();\n")

                # Copy op call lines until ");"
                while i < len(lines):
                    output.append(lines[i])
                    if lines[i].rstrip().endswith(");"):
                        i += 1
                        break
                    i += 1

                # INSTR_END + close block
                output.append(f"{indent}INSTR_END({INSTR_ARGS}); }}\n")
                n_instrumented += 1
                continue
            else:
                output.append(line)
                i += 1
                continue
        else:
            output.append(line)
            i += 1

    # Validation
    if not header_inserted:
        print("WARNING: 'using namespace cute;' not found — #include not inserted")
    if not global_shared_inserted:
        print("WARNING: __global__ function not found — static SMEM not inserted")

    # ---- Second pass: insert __shared__ InstrTrace AFTER last kernel __shared__ ----
    # This places the trace buffer at the END of shared memory layout.
    last_shared_idx = None
    for idx, line in enumerate(output):
        if "__shared__" in line and "InstrTrace" not in line:
            last_shared_idx = idx
    if last_shared_idx is not None:
        # Insert after the last kernel __shared__ declaration
        indent = global_fn_indent
        output.insert(last_shared_idx + 1,
                       f"{indent}__shared__ InstrTrace _instr_buf[MAX_INSTR_ENTRIES];\n")
    else:
        # No kernel __shared__ found — insert at original position (after _instr_idx)
        for idx, line in enumerate(output):
            if "int _instr_idx = 0;" in line:
                indent = global_fn_indent
                output.insert(idx,
                               f"{indent}__shared__ InstrTrace _instr_buf[MAX_INSTR_ENTRIES];\n")
                break

    with open(output_path, "w") as f:
        f.writelines(output)

    # Write metadata JSON
    meta_path = output_path + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_sequence, f, indent=1)

    print(f"Instrumented {n_instrumented} op calls.")
    print(f"Written to {output_path}")
    print(f"Metadata: {meta_path} ({len(meta_sequence)} entries)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_kernel.inc> <output_kernel_instrumented.inc>")
        sys.exit(1)
    instrument(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
