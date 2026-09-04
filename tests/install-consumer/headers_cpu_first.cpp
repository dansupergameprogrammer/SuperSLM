#include <superslm/sslm_abi.h>
#include <superslm/gpu_1p0.h>

#include <type_traits>

static_assert(std::is_enum_v<sslm_status>);
static_assert(std::is_enum_v<SslmGpuStatus>);
static_assert(!std::is_convertible_v<SslmGpuStatus, unsigned int>);

int main() {
	const sslm_status cpu = SSLM_OK;
	const SslmGpuStatus gpu = SslmGpuStatus::SSLM_OK;
	return cpu == SSLM_OK && gpu == SslmGpuStatus::SSLM_OK ? 0 : 1;
}
