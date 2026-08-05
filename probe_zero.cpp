// Lesser plane, executed for accuracy only (not prosecuted): what does the
// reciprocal door return at the degenerate mantissas the int32 check admits?
#include <cstdio>
#include <cstdint>
#include "superslm/checked_chain_funnel.h"
using namespace superslm;
int main() {
	std::printf("CarriedScaleReciprocal(0)  = %lld\n", (long long)CarriedScaleReciprocal(0));
	std::printf("CarriedScaleReciprocal(-2) = %lld\n", (long long)CarriedScaleReciprocal(-2));
	std::printf("CarriedScaleReciprocal(-1073741824) = %lld\n",
	            (long long)CarriedScaleReciprocal(-(int64_t{1} << 30)));
	return 0;
}
