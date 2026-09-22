// T-2948 round 2 / F1: the production ShaderPath staleness gate in a developer tree
// whose source directory has an ANSI non-UTF-8 byte. No GPU device is created.
#include "../../src/gpu/d3d12_harness.h"

#include <windows.h>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
namespace harness = superslm_gpu::harness;

static int checks = 0;
static int failures = 0;
#define CHECK_F1(cond, message) do { ++checks; if (!(cond)) { ++failures; \
	std::printf("FAIL F1: %s\n", message); } } while (0)

static std::string ToMulti(UINT page, const std::wstring& wide) {
	const int n = WideCharToMultiByte(page, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
	if (n <= 0) return {};
	std::string result(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(page, 0, wide.c_str(), -1, result.data(), n, nullptr, nullptr);
	result.pop_back();
	return result;
}

static uint64_t WriteTime(const fs::path& path) {
	WIN32_FILE_ATTRIBUTE_DATA data{};
	if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data)) return 0;
	ULARGE_INTEGER t{};
	t.LowPart = data.ftLastWriteTime.dwLowDateTime;
	t.HighPart = data.ftLastWriteTime.dwHighDateTime;
	return t.QuadPart;
}

int main(int argc, char** argv) {
	if (argc != 2 || (std::string(argv[1]) != "baseline" &&
	                  std::string(argv[1]) != "override")) {
		std::printf("usage: cell_shader_stale_unicode.exe baseline|override\n");
		return 2;
	}
	wchar_t module[MAX_PATH]{};
	const DWORD n = GetModuleFileNameW(nullptr, module, MAX_PATH);
	CHECK_F1(n > 0 && n < MAX_PATH, "cannot resolve executable path");
	if (failures) return 2;
	const fs::path tree = fs::path(module).parent_path().parent_path().parent_path();
	const fs::path source = tree / "src" / "gpu" / "shaders" / "dyn_recip.hlsl";
	const fs::path cso = tree / "compiled" / "dyn_recip.cso";
	const std::string& source_dir_ansi = harness::ShaderSourceDirOrEmpty();
	CHECK_F1(!source_dir_ansi.empty(), "developer source directory was not discovered");
	CHECK_F1(GetACP() != CP_UTF8, "test requires an ANSI code page distinct from UTF-8");
	const int utf8_source_length = MultiByteToWideChar(
	    CP_UTF8, MB_ERR_INVALID_CHARS, source_dir_ansi.c_str(), -1, nullptr, 0);
	CHECK_F1(utf8_source_length == 0, "source path ANSI bytes unexpectedly decode as UTF-8");
	const uint64_t source_time = WriteTime(source);
	const uint64_t cso_time = WriteTime(cso);
	CHECK_F1(source_time != 0 && cso_time != 0 && source_time > cso_time,
	         "fixture must contain a source newer than the binary");
	if (failures) {
		std::printf("arm=%s checks=%d failures=%d\n", argv[1], checks, failures);
		return 2;
	}
	const std::string cso_ansi = ToMulti(CP_ACP, cso.wstring());
	const std::string cso_utf8 = ToMulti(CP_UTF8, cso.wstring());
	if (std::string(argv[1]) == "baseline") {
		const std::string diagnostic = harness::ShaderBinaryStalenessDiagnostic(
		    source_dir_ansi, "dyn_recip", cso_ansi);
		CHECK_F1(diagnostic.find("stale shader binary") != std::string::npos,
		         "default ANSI path did not reject the stale binary");
	} else {
		std::wstring normalized;
		const std::string override_utf8 = ToMulti(CP_UTF8, cso.parent_path().wstring());
		CHECK_F1(harness::CheckShaderDirOverride(override_utf8.c_str(), &normalized) ==
		             harness::ShaderDirCheck::Ok,
		         "pre-device override validation rejected the fixture directory");
		if (!failures) {
			CHECK_F1(harness::CommitShaderDirOverride(normalized) == harness::ShaderDirCheck::Ok,
			         "override state could not be fixed");
			CHECK_F1(harness::ShaderDirOverrideActive(), "override did not become active");
			const std::string diagnostic = harness::ShaderBinaryStalenessDiagnostic(
			    source_dir_ansi, "dyn_recip", cso_utf8);
			CHECK_F1(diagnostic.find("stale shader binary") != std::string::npos,
			         "override decoded the ANSI source path as UTF-8 and missed stale .cso");
			bool refused = false;
			try {
				(void)harness::ShaderPath("dyn_recip");
			} catch (const std::exception& e) {
				refused = std::string(e.what()).find("stale shader binary") != std::string::npos;
			}
			CHECK_F1(refused, "production ShaderPath returned a stale override binary");
		}
	}
	std::printf("arm=%s acp=%u source_100ns=%llu cso_100ns=%llu checks=%d failures=%d\n",
	            argv[1], GetACP(), static_cast<unsigned long long>(source_time),
	            static_cast<unsigned long long>(cso_time), checks, failures);
	return failures ? 1 : 0;
}
