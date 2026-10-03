// T-3103 S0-SF1 reference harness (plan rev 7 section 6 row 12).
//
// Drives the real SuperFAISS master ScratchBank::Create -> Append -> Snapshot -> Query seam
// and prints a JSON receipt of what the library actually did. It decides nothing: the
// independent checker sf1_check.py compares the receipt against sf1-fixture.json.
//
// Input: a cells file written by run_sf1.ps1 from sf1-fixture.json:
//   CELLS <count>
//   CELL <name> <dims> <capacity>
//   ROW <hex32 x dims>      (capacity lines, appended in order)
//   PROBE <hex32 x dims>    (capacity lines)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "superfaiss/alloc.h"
#include "superfaiss/query.h"
#include "superfaiss/scratch.h"
#include "superfaiss/types.h"

using namespace superfaiss;

namespace
{
const char* StatusName(Status s)
{
	switch (s)
	{
	case Status::Ok: return "Ok";
	case Status::InvalidArgument: return "InvalidArgument";
	case Status::DimsMismatch: return "DimsMismatch";
	case Status::NonFiniteQuery: return "NonFiniteQuery";
	case Status::ZeroNormQuery: return "ZeroNormQuery";
	case Status::NonZeroPadding: return "NonZeroPadding";
	case Status::BadAlignment: return "BadAlignment";
	case Status::BadFormat: return "BadFormat";
	case Status::ZeroNormRow: return "ZeroNormRow";
	case Status::OutOfMemory: return "OutOfMemory";
	}
	return "Unknown";
}

uint32_t Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float FromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

[[noreturn]] void Die(const char* why)
{
	std::fprintf(stderr, "HARNESS ERROR: %s\n", why);
	std::exit(2);
}

std::vector<float> ReadVec(std::ifstream& in, const char* tag, int dims)
{
	std::string word;
	if (!(in >> word) || word != tag) Die("cells file: unexpected tag");
	std::vector<float> v(static_cast<size_t>(dims));
	for (int i = 0; i < dims; ++i)
	{
		std::string hex;
		if (!(in >> hex)) Die("cells file: short row");
		v[static_cast<size_t>(i)] = FromBits(static_cast<uint32_t>(std::strtoul(hex.c_str(), nullptr, 16)));
	}
	return v;
}

void RunCell(std::ifstream& in, bool last)
{
	std::string word, name;
	int dims = 0, capacity = 0;
	if (!(in >> word >> name >> dims >> capacity) || word != "CELL") Die("cells file: CELL");
	std::vector<std::vector<float>> rows, probes;
	for (int i = 0; i < capacity; ++i) rows.push_back(ReadVec(in, "ROW", dims));
	for (int i = 0; i < capacity; ++i) probes.push_back(ReadVec(in, "PROBE", dims));

	ScratchBank bank;
	const Status created = bank.Create(capacity, dims, Metric::Dot, Quantization::Float32);
	std::printf("  {\"cell\":\"%s\",\"create\":\"%s\",\"padded_dims\":%d,\"appends\":[", name.c_str(),
		StatusName(created), bank.GetPaddedDims());
	for (int i = 0; i < capacity; ++i)
	{
		int32_t out = -1;
		const Status s = bank.Append(rows[static_cast<size_t>(i)].data(), dims, &out);
		std::printf("%s{\"status\":\"%s\",\"out_index\":%d}", i ? "," : "", StatusName(s), out);
	}
	BankView view;
	std::vector<uint32_t> tomb(static_cast<size_t>(ScratchBank::TombstoneWords(capacity)) + 1, 0u);
	const Status snap = bank.Snapshot(&view, tomb.data());
	std::printf("],\"snapshot\":\"%s\",\"count\":%d,\"view_dims\":%d,\"view_padded\":%d,\"row_bytes\":%lld,\"readback\":[",
		StatusName(snap), view.count, view.dims, view.paddedDims,
		static_cast<long long>(view.count) * view.paddedDims * 4);
	const float* base = static_cast<const float*>(view.rows);
	for (int r = 0; r < view.count; ++r)
	{
		std::printf("%s[", r ? "," : "");
		for (int c = 0; c < view.paddedDims; ++c)
			std::printf("%s\"%08x\"", c ? "," : "", Bits(base[static_cast<int64_t>(r) * view.paddedDims + c]));
		std::printf("]");
	}
	std::printf("],\"tombstone_words\":[");
	for (int w = 0; w < ScratchBank::TombstoneWords(view.count); ++w)
		std::printf("%s%u", w ? "," : "", tomb[static_cast<size_t>(w)]);
	std::printf("],\"queries\":[");
	Workspace ws;
	for (int q = 0; q < capacity; ++q)
	{
		std::vector<float, std::allocator<float>> storage(static_cast<size_t>(view.paddedDims) + 8, 0.0f);
		float* aligned = storage.data();
		while (reinterpret_cast<uintptr_t>(aligned) % kAlignment != 0) ++aligned;
		std::memcpy(aligned, probes[static_cast<size_t>(q)].data(), static_cast<size_t>(dims) * sizeof(float));
		QueryParams params;
		params.k = 1;
		params.excludeBits = tomb.data();
		Hit hit;
		int32_t hitCount = 0;
		const Status qs = Query(view, aligned, params, ws, &hit, &hitCount);
		std::printf("%s{\"status\":\"%s\",\"hit_count\":%d,\"index\":%d,\"score_bits\":\"%08x\"}", q ? "," : "",
			StatusName(qs), hitCount, hit.index, Bits(hit.score));
	}
	std::printf("]}%s\n", last ? "" : ",");
}

void Observe(const char* name, const std::vector<std::vector<float>>& rows, const std::vector<int>& dimsArg,
	int capacity, bool last)
{
	ScratchBank bank;
	const Status created = bank.Create(capacity, 3, Metric::Dot, Quantization::Float32);
	std::printf("  {\"observation\":\"%s\",\"create\":\"%s\",\"statuses\":[", name, StatusName(created));
	for (size_t i = 0; i < rows.size(); ++i)
	{
		int32_t out = -1;
		const Status s = bank.Append(rows[i].data(), dimsArg[i], &out);
		std::printf("%s\"%s\"", i ? "," : "", StatusName(s));
	}
	std::printf("],\"count_after\":%d}%s\n", bank.Count(), last ? "" : ",");
}
} // namespace

int main(int argc, char** argv)
{
	if (argc != 2) Die("usage: sf1_harness <cells-file>");
	std::ifstream in(argv[1]);
	if (!in) Die("cannot open cells file");
	std::string word;
	int cells = 0;
	if (!(in >> word >> cells) || word != "CELLS" || cells <= 0) Die("cells file: CELLS");
	std::printf("{\"harness\":\"T-3103 sf1_harness\",\"cells\":[\n");
	for (int c = 0; c < cells; ++c) RunCell(in, c + 1 == cells);
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const std::vector<float> unit = {1.0f, 0.0f, 0.0f}, bad = {0.0f, nan, 0.0f};
	std::printf("],\"observations\":[\n");
	Observe("OBS-WRONG-DIMS", {unit}, {2}, 3, false);
	Observe("OBS-NONFINITE", {bad}, {3}, 3, false);
	Observe("OBS-PARTIAL", {unit, bad}, {3, 3}, 3, false);
	Observe("OBS-OVER-CAPACITY", {unit, unit, unit, unit}, {3, 3, 3, 3}, 3, true);
	std::printf("]}\n");
	return 0;
}
