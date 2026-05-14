"""Probe OpenVINO NPU capability and CACHE_MODE support."""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from ctypes import wintypes


def _safe_dist_version(dist_name: str) -> str:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return "<not installed>"
    except Exception as exc:  # pragma: no cover
        return f"<error: {exc}>"


def _safe_module_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "<no __version__>"))
    except Exception as exc:
        return f"<import failed: {exc}>"


def print_versions() -> None:
    print("=== Environment ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    uname = platform.uname()
    print(f"Machine: {uname.machine}")
    print(f"Processor: {platform.processor() or '<unknown>'}")
    print(f"CPU logical cores: {os.cpu_count()}")
    print(f"Hostname: {uname.node}")
    print()
    print("=== Package Versions ===")
    print(f"openvino (dist): {_safe_dist_version('openvino')}")
    print(f"openvino (module): {_safe_module_version('openvino')}")
    print(f"openvino-genai (dist): {_safe_dist_version('openvino-genai')}")
    print(f"openvino_genai (module): {_safe_module_version('openvino_genai')}")
    print(f"wheel (dist): {_safe_dist_version('wheel')}")


def _safe_get_property(core, device: str, key: str) -> str:
    try:
        value = core.get_property(device, key)
        return str(value)
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _safe_version_build(version_obj) -> str:
    build_number = getattr(version_obj, "build_number", None)
    if build_number:
        return str(build_number)
    legacy_build = getattr(version_obj, "buildNumber", None)
    if legacy_build:
        return str(legacy_build)
    return "<unknown>"


def _safe_plugin_build(core, device: str) -> str:
    try:
        versions = core.get_versions(device)
        version_obj = versions.get(device)
        if version_obj is None and versions:
            version_obj = next(iter(versions.values()))
        if version_obj is None:
            return "<unknown>"
        return _safe_version_build(version_obj)
    except Exception as exc:
        return f"<unavailable: {exc}>"


def print_device_inventory(core) -> None:
    print()
    print("=== Device Inventory ===")
    if not core.available_devices:
        print("No OpenVINO devices found.")
        return

    for dev in core.available_devices:
        full_name = _safe_get_property(core, dev, "FULL_DEVICE_NAME")
        plugin_version = _safe_plugin_build(core, dev)
        print(f"{dev}:")
        print(f"  FULL_DEVICE_NAME: {full_name}")
        print(f"  Plugin build: {plugin_version}")


def print_runtime_details(core, device: str) -> None:
    print()
    print("=== OpenVINO Runtime ===")
    print(f"OpenVINO runtime version: {_safe_plugin_build(core, device)}")
    print(f"Available devices: {', '.join(core.available_devices)}")
    print(f"Probe device: {device}")
    print(f"FULL_DEVICE_NAME: {_safe_get_property(core, device, 'FULL_DEVICE_NAME')}")
    print(f"NPU_DRIVER_VERSION: {_safe_get_property(core, device, 'NPU_DRIVER_VERSION')}")
    print(f"NPU_COMPILER_TYPE: {_safe_get_property(core, device, 'NPU_COMPILER_TYPE')}")
    print(f"NPU_COMPILER_VERSION: {_safe_get_property(core, device, 'NPU_COMPILER_VERSION')}")


@dataclass
class ProbeResult:
    success: bool
    classification: str
    error_text: str = ""


def probe_cache_mode(core, device: str, cache_dir: str | None, cache_mode: str) -> ProbeResult:
    import openvino as ov

    # Minimal single-op model to test compile-time option acceptance.
    param = ov.opset13.parameter([1, 80, 16], ov.Type.f32, name="x")
    relu = ov.opset13.relu(param)
    model = ov.Model([relu], [param], "cache_mode_probe")

    config: dict[str, str] = {"CACHE_MODE": cache_mode}
    if cache_dir:
        config["CACHE_DIR"] = cache_dir

    try:
        core.compile_model(model, device, config)
        return ProbeResult(success=True, classification="accepted")
    except Exception as exc:  # pylint: disable=broad-except
        text = str(exc)
        if "Option 'CACHE_MODE' is not supported" in text or "CACHE_MODE is not supported" in text:
            return ProbeResult(
                success=False,
                classification="unsupported_option",
                error_text=text,
            )
        return ProbeResult(
            success=False,
            classification="other_error",
            error_text=text,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print OpenVINO/OpenVINO GenAI versions and probe whether "
            "the selected device accepts CACHE_MODE."
        )
    )
    parser.add_argument(
        "--device",
        "-d",
        default="NPU",
        help="OpenVINO device string (default: NPU)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional cache directory passed as CACHE_DIR",
    )
    parser.add_argument(
        "--cache_mode",
        default="OPTIMIZE_SIZE",
        help="CACHE_MODE value to probe (default: OPTIMIZE_SIZE)",
    )
    return parser.parse_args()


def ensure_cache_dir(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None

    path = Path(cache_dir).expanduser()
    try:
        existed_before = path.exists()
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create cache directory '{path}': {exc}") from exc

    if existed_before:
        print(f"Cache directory exists: {path}")
    else:
        print(f"Created cache directory: {path}")
    return str(path)


def _enumerate_loaded_modules_windows() -> list[Path]:
    if platform.system().lower() != "windows":
        return []

    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    hmodule = wintypes.HMODULE
    lpdword = ctypes.POINTER(wintypes.DWORD)

    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(hmodule),
        wintypes.DWORD,
        lpdword,
        wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        hmodule,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    list_modules_all = 0x03

    arr_size = 256
    module_handle_size = ctypes.sizeof(hmodule)
    while True:
        modules = (hmodule * arr_size)()
        needed = wintypes.DWORD()
        ok = psapi.EnumProcessModulesEx(
            process,
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
            list_modules_all,
        )
        if not ok:
            return []

        needed_count = needed.value // module_handle_size
        if needed_count <= arr_size:
            break
        arr_size = needed_count + 64

    results: list[Path] = []
    path_buffer = ctypes.create_unicode_buffer(32768)
    for i in range(needed_count):
        path_buffer.value = ""
        written = psapi.GetModuleFileNameExW(
            process,
            modules[i],
            path_buffer,
            len(path_buffer),
        )
        if written:
            try:
                results.append(Path(path_buffer.value))
            except Exception:
                continue
    return results


def print_loaded_npu_dll_paths(core, device: str) -> None:
    print()
    print("=== Loaded NPU DLLs ===")

    module_paths = _enumerate_loaded_modules_windows()
    if not module_paths:
        print("Could not enumerate process modules (or unsupported OS).")
        return

    targets = {
        "openvino_intel_npu_plugin.dll",
        "openvino_intel_npu_compiler.dll",
    }
    found: dict[str, Path] = {}
    for module_path in module_paths:
        name = module_path.name.lower()
        if name in targets and name not in found:
            found[name] = module_path

    plugin_name = "openvino_intel_npu_plugin.dll"
    compiler_name = "openvino_intel_npu_compiler.dll"
    plugin_path = found.get(plugin_name)
    compiler_path = found.get(compiler_name)

    if plugin_path is not None:
        print(f"{plugin_name}:")
        print(f"  Path: {plugin_path}")
        print(f"  Directory: {plugin_path.parent}")
    else:
        print(f"{plugin_name}: <not found in loaded module list>")

    if compiler_path is not None:
        print(f"{compiler_name}:")
        print(f"  Path: {compiler_path}")
        print(f"  Directory: {compiler_path.parent}")
    else:
        print(f"{compiler_name}: <not found in loaded module list>")

    if plugin_path is not None and compiler_path is not None:
        same_dir = plugin_path.parent.resolve() == compiler_path.parent.resolve()
        print(f"Same directory: {same_dir}")
    else:
        print(
            "Note: one or both DLLs were not found. "
            "This can happen with delayed loading; rerun after NPU compile/probe."
        )

    if plugin_path is not None:
        sibling_compiler = plugin_path.parent / compiler_name
        print(
            f"Sibling compiler DLL exists next to plugin: {sibling_compiler.exists()}"
        )
        print(f"Sibling compiler path: {sibling_compiler}")

    compiler_type = _safe_get_property(core, device, "NPU_COMPILER_TYPE")
    compiler_type_text = compiler_type.upper()
    backend_summary = "UNKNOWN"
    if "DRIVER" in compiler_type_text:
        backend_summary = "DRIVER"
    elif "PLUGIN" in compiler_type_text:
        backend_summary = "PLUGIN"
    print(f"Compilation backend summary: {backend_summary} ({compiler_type})")


def main() -> int:
    args = parse_args()
    print_versions()

    try:
        import openvino as ov
    except Exception as exc:
        print()
        print("Failed to import openvino runtime.")
        print(f"Error: {exc}")
        return 2

    core = ov.Core()
    print_device_inventory(core)
    device = args.device.upper()
    if device not in core.available_devices:
        print()
        print(f"Requested device '{device}' is not available.")
        print(f"Available devices: {', '.join(core.available_devices)}")
        return 3

    print_runtime_details(core, device)

    try:
        cache_dir = ensure_cache_dir(args.cache_dir)
    except RuntimeError as exc:
        print()
        print(str(exc))
        return 5

    print()
    print("=== CACHE_MODE Probe ===")
    print(f"Requested CACHE_MODE: {args.cache_mode}")
    print(f"CACHE_DIR: {cache_dir if cache_dir else '<not set>'}")
    result = probe_cache_mode(core, device, cache_dir, args.cache_mode)

    print(f"Result: {'SUCCESS' if result.success else 'FAIL'}")
    print(f"Classification: {result.classification}")
    if result.error_text:
        print("Error:")
        print(result.error_text)

    # Enumerate loaded modules after runtime init and probe call to capture delayed loads.
    print_loaded_npu_dll_paths(core, device)

    # Keep unsupported-option as informative outcome for diagnostics.
    if result.classification == "unsupported_option":
        return 0
    return 0 if result.success else 4


if __name__ == "__main__":
    raise SystemExit(main())
