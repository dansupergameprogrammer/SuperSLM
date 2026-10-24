// Precached prefixes, the F2 cross-card leg (plan §4 item 3 step 6, on two certified GPUs): a GPU prefix blob saved
// on one card is refused SSLM_RESTORE_PRODUCER_MISMATCH on the other, with a null out-handle, and restores on its own.
// pcg_cell_format proves the refusal on one card by editing the blob's adapter fields; this carries a real blob
// between the two cards (the RTX 2080 SUPER, vendor 10de, and the RX 7900 XTX, vendor 1002).
//
// Run (adapter chosen by SSLM_GPU_ADAPTER_INDEX, as every cell):
//   pcg_cross_card.exe --fixture-dir=DIR --save=FILE      builds a 40-token frozen prefix of pkv_def, saves it, writes
//                                                         FILE, and checks it restores on this card
//   pcg_cross_card.exe --fixture-dir=DIR --restore=FILE   restores FILE: SSLM_RESTORE_PRODUCER_MISMATCH and a null
//                                                         out-handle when the blob names another adapter, SSLM_OK when
//                                                         it names this one
// Both may be given in one run; the save happens first.

#include "pcg_helpers.h"

#include <fstream>
#include <iterator>

using namespace pcg;

namespace {

std::string ArgValue(int argc, char** argv, const char* name) {
	const std::string prefix = std::string(name) + "=";
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
	}
	return std::string();
}

uint32_t Read32(const std::vector<uint8_t>& b, size_t off) {
	uint32_t v = 0;
	if (b.size() >= off + 4) std::memcpy(&v, b.data() + off, 4);
	return v;
}

void Save(const Env& e, const std::string& file) {
	Rig rig(e, 32);
	if (!rig.pool) return;
	std::vector<int32_t> tokens;
	for (int32_t t = 1; t <= 40; ++t) tokens.push_back(t);
	SslmGpuPrefixHandle* p = Prefix(rig, tokens);
	if (!p) return;
	const std::vector<uint8_t> blob = SavePrefix(e, p);
	CHECK_MSG(!blob.empty(), "save: no blob");
	if (blob.empty()) return;
	std::ofstream out(file, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
	out.close();
	CHECK_MSG(out.good(), "save: could not write %s", file.c_str());
	std::printf("saved %zu bytes to %s: producer vendor %04x device %04x, engine %u.%u.%u\n", blob.size(), file.c_str(),
	            Read32(blob, kOffVendor), Read32(blob, kOffDevice), Read32(blob, kOffMajor), Read32(blob, kOffMinor),
	            Read32(blob, kOffPatch));
	const RestoredPrefix same = RestorePrefix(rig, blob);
	CHECK_MSG(same.status == GS::SSLM_OK && same.p != nullptr, "save: the blob restores on the card that saved it (%s)",
	          Name(same.status));
}

void Restore(const Env& e, const std::string& file) {
	std::ifstream in(file, std::ios::binary);
	const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	CHECK_MSG(blob.size() >= kHeader, "restore: %s is missing or shorter than the SPX1 header (%zu bytes)", file.c_str(),
	          blob.size());
	if (blob.size() < kHeader) return;
	uint32_t vendor = 0, device = 0;
	const bool have_ids = AdapterIds(&vendor, &device);
	CHECK_MSG(have_ids, "restore: this adapter's ids could not be read");
	if (!have_ids) return;
	const uint32_t bv = Read32(blob, kOffVendor), bd = Read32(blob, kOffDevice);
	Rig rig(e, 32);
	if (!rig.pool) return;
	const RestoredPrefix r = RestorePrefix(rig, blob);
	if (bv != vendor || bd != device) {
		std::printf("restore: blob from vendor %04x device %04x, this adapter vendor %04x device %04x: another card\n", bv, bd,
		            vendor, device);
		Refused(r, kProducerMismatch, "restore of another card's blob");
	} else {
		std::printf("restore: blob from this card (vendor %04x device %04x)\n", bv, bd);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p != nullptr, "restore of this card's own blob: %s", Name(r.status));
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	const std::string save = ArgValue(argc, argv, "--save");
	const std::string restore = ArgValue(argc, argv, "--restore");
	CHECK_MSG(!save.empty() || !restore.empty(), "pass --save=FILE, --restore=FILE, or both");
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			if (!save.empty()) Save(e, save);
			if (!restore.empty()) Restore(e, restore);
		}
	}
	return FinishSuite("pcg_cross_card");
}
