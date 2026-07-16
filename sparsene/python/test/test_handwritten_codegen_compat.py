import builtins
import io
from pathlib import Path
import re
import runpy
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTBED_ROOT = REPO_ROOT / "examples" / "src_fp32"
CASES = (
    ("acc", "acc_ops.py", "kernel.inc"),
    ("acc", "acc_ops_spmv.py", "kernel_spmv.inc"),
    ("bitbsr", "bitbsr_ops.py", "kernel.inc"),
    ("bitbsr", "bitbsr_ops_spmv.py", "kernel_spmv.inc"),
    ("dtc", "dtc_ops.py", "kernel.inc"),
    ("dtc", "dtc_ops_spmv.py", "kernel_spmv.inc"),
    ("dtc", "dtc_ops_multi_binding.py", "kernel_multi_binding.inc"),
    ("dtc", "dtc_ops_strict_lb.py", "kernel_strict_lb.inc"),
    ("sr_bcrs", "sr_bcrs_ops.py", "kernel.inc"),
    ("sr_bcrs", "sr_bcrs_ops_16x8_spmv.py", "kernel_16x8_spmv.inc"),
)


class _GeneratedSource(io.StringIO):
    def close(self):
        # runpy executes ``with open(...):``; retain the generated source.
        pass


def _render_handwritten_program(format_name: str, script_name: str) -> str:
    script = TESTBED_ROOT / format_name / "testbed" / script_name
    generated = _GeneratedSource()
    real_open = builtins.open

    def redirected_open(file, mode="r", *args, **kwargs):
        if Path(file).suffix == ".inc" and "w" in mode:
            return generated
        return real_open(file, mode, *args, **kwargs)

    with mock.patch("builtins.open", redirected_open):
        runpy.run_path(str(script), run_name="__main__")
    return generated.getvalue()


def _remove_dtc_instrumentation(source: str) -> str:
    source = re.sub(
        r"__device__\s+__forceinline__\s+unsigned\s+int\s+get_smid_ptx\(\)\s*\{.*?\}\s*",
        "",
        source,
        flags=re.DOTALL,
    )
    source = re.sub(
        r"unsigned\s+long\s+long\s*\*\s*dsm_cycles\s*,\s*int\s*\*\s*dblock_smid\s*,",
        "",
        source,
    )
    source = re.sub(
        r"constexpr\s+int\s+kTileB\s*=.*?"
        r"static_assert\(\(kTileB\s*%\s*8\)\s*==\s*0,.*?\);",
        "",
        source,
        flags=re.DOTALL,
    )
    return source


def _normalize_cuda(source: str, *, format_name: str) -> str:
    if format_name == "dtc":
        source = _remove_dtc_instrumentation(source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r"\s+", "", source)


def test_handwritten_nvir_codegen_matches_existing_inc():
    for format_name, script_name, inc_name in CASES:
        generated = _render_handwritten_program(format_name, script_name)
        expected_path = TESTBED_ROOT / format_name / "testbed" / inc_name
        expected = expected_path.read_text()

        assert "int tid, lid, wid;" in generated
        assert _normalize_cuda(
            generated, format_name=format_name
        ) == _normalize_cuda(expected, format_name=format_name), format_name
