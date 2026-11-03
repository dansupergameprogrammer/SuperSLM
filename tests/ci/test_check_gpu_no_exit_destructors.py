"""Red suite for tests/ci/check_gpu_no_exit_destructors.py: fixture cells for the mechanism, then the real tree."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_gpu_no_exit_destructors as m  # noqa: E402


def found(text):
    return m.check_text("x.h", text)


def test_function_local_comptr_by_value_is_reported():
    assert found("void f() {\n\tstatic ComPtr<ID3D12PipelineState> s_pso;\n}\n")


def test_function_local_map_of_pipelines_is_reported():
    assert found("CachedPipeline& g() {\n\tstatic std::map<std::string, CachedPipeline> cache;\n}\n")


def test_namespace_inline_holder_by_value_is_reported():
    assert found("inline DeviceHolder g_device_holder;\n") == []  # the holder itself is checked by its union rule
    assert found("inline Device g_device;\n")


def test_never_destroyed_references_pass():
    assert found("void f() {\n\tstatic auto& s_pso = *new ComPtr<ID3D12PipelineState>();\n"
                 "\tstatic auto& cache = *new std::map<std::string, CachedPipeline>();\n}\n") == []


def test_unrelated_statics_and_comments_pass():
    assert found("static std::atomic<uint64_t> g_count{0};\n// static ComPtr<X> s_old;\n"
                 "static const char* k = \"static ComPtr<X> y;\";\nstatic Device& GetDevice();\n") == []


def test_holder_with_device_by_value_is_reported():
    assert found("struct DeviceHolder {\n\tstd::mutex mutex;\n\tDevice device;\n\tbool a = false;\n};\n")


def test_holder_with_device_in_union_passes():
    assert found("struct DeviceHolder {\n\tstd::mutex mutex;\n\tunion {\n\t\tDevice device;\n\t};\n"
                 "\tDeviceHolder() noexcept : device() {}\n\t~DeviceHolder() {}\n};\n") == []


def test_real_tree_passes():
    assert m.scan(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))) == []
