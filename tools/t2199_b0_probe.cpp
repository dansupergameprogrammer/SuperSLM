// T-2199 Phase B0 calibration probe.
//
// Replays captured greedy logit rows through the shipped Phase A/C primitives. The replay is
// deliberately counterfactual: the anti-LM follows the recorded greedy prefix, so divergence is
// exact up to the first changed token and is reported as reach/effectiveness ceiling, never as an
// achieved autoregressive score. Candidate rows are confirmed later through sslm_generate's real
// damped-greedy loop.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/intmath.h"
#include "superslm/sslm_damped_greedy.h"

namespace {

using superslm::AntiLmState;

constexpr int64_t kQLn2 = 493;
constexpr int64_t kQB = 964;
constexpr int64_t kQC = 487361;
constexpr double kAlphas[] = {0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 30.0};
constexpr int kOrders[] = {1, 2, 3};
constexpr int kWidths[] = {3, 6, 10};

struct Dump {
	uint64_t rows = 0;
	uint64_t vocab = 0;
	std::vector<int32_t> logits;
};

bool LoadDump(const char* path, Dump* out) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) return false;
	const std::streamoff bytes = f.tellg();
	f.seekg(0);
	f.read(reinterpret_cast<char*>(&out->rows), sizeof(out->rows));
	f.read(reinterpret_cast<char*>(&out->vocab), sizeof(out->vocab));
	if (!f || out->rows == 0 || out->vocab == 0 ||
	    out->rows > std::numeric_limits<size_t>::max() / out->vocab) return false;
	const size_t count = static_cast<size_t>(out->rows * out->vocab);
	const uint64_t expected = 16ull + static_cast<uint64_t>(count) * sizeof(int32_t);
	if (bytes < 0 || static_cast<uint64_t>(bytes) != expected) return false;
	out->logits.resize(count);
	f.read(reinterpret_cast<char*>(out->logits.data()),
	       static_cast<std::streamsize>(count * sizeof(int32_t)));
	return static_cast<bool>(f);
}

std::string BaseName(const char* path) {
	std::string s(path);
	const size_t slash = s.find_last_of("\\/");
	return slash == std::string::npos ? s : s.substr(slash + 1);
}

int64_t Median(std::vector<int64_t> values) {
	if (values.empty()) return 0;
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

bool IExpValue(int64_t shifted, int64_t m, int64_t* out) {
	superslm::IExpConstruction c{};
	const superslm::IExpDomain d = superslm::IExpConstruct(shifted, kQLn2, kQB, kQC, &c);
	if (d != superslm::IExpDomain::kOk) return false;
	const int64_t value = superslm::IExpEvaluate(c);
	if (value < 0 || value > m) return false;
	*out = value;
	return true;
}

bool FullRowZ(const int32_t* row, size_t vocab, std::vector<int64_t>* wide,
	          std::vector<int64_t>* shifted, int64_t* out_z) {
	wide->resize(vocab);
	shifted->resize(vocab);
	for (size_t i = 0; i < vocab; ++i) (*wide)[i] = row[i];
	superslm::ShiftByMax(wide->data(), vocab, shifted->data());
	const int64_t m = kQB * kQB + kQC;
	int64_t total = 0;
	for (size_t i = 0; i < vocab; ++i) {
		int64_t value = 0;
		if (!IExpValue((*shifted)[i], m, &value) ||
		    total > std::numeric_limits<int64_t>::max() - value) return false;
		total += value;
	}
	*out_z = total;
	return total > 0;
}

int32_t Select(const int32_t* candidates, const int64_t* q, const int64_t* p, int k,
	           int64_t alpha_q15) {
	int32_t best_token = -1;
	int64_t best_score = 0;
	bool have_best = false;
	for (int i = 0; i < k; ++i) {
		const int64_t penalty = (alpha_q15 * p[i]) >> superslm::kProbFracBits;
		const int64_t score = q[i] - penalty;
		if (!have_best || score > best_score ||
		    (score == best_score && candidates[i] < best_token)) {
			best_token = candidates[i];
			best_score = score;
			have_best = true;
		}
	}
	return best_token;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: t2199_b0_probe <out.tsv> <dump.logits>...\n");
		return 2;
	}
	std::ofstream out(argv[1], std::ios::binary);
	if (!out) {
		std::fprintf(stderr, "cannot open output: %s\n", argv[1]);
		return 3;
	}
	out << "META\tq_ln2\t" << kQLn2 << "\tq_b\t" << kQB << "\tq_c\t" << kQC
	    << "\tengine\tproduction-phase-a-c\n";
	out << "A\tfile\talpha\tn\tk\tsteps\tndiv\tfirst_div\n";
	out << "S\tfile\tk\tsteps\tpmin\tpmax\tdispersion\tmedian_qspread\n";
	out << "SN\tfile\tn\tk\tsteps\tmedian_pomspread\n";
	out << "C\tfile\tk\tmean_ns\trepetitions\n";

	for (int file_index = 2; file_index < argc; ++file_index) {
		Dump dump;
		if (!LoadDump(argv[file_index], &dump) || dump.vocab > INT32_MAX) {
			std::fprintf(stderr, "invalid dump: %s\n", argv[file_index]);
			return 4;
		}
		const std::string name = BaseName(argv[file_index]);
		const int32_t vocab = static_cast<int32_t>(dump.vocab);
		std::vector<uint8_t> mask((dump.vocab + 7) / 8, 0xFF);
		std::vector<int32_t> top10(10);
		std::vector<int64_t> wide, shifted;
		AntiLmState* anti[3] = {
		    superslm::AntiLmCreate(1), superslm::AntiLmCreate(2), superslm::AntiLmCreate(3)};
		if (!anti[0] || !anti[1] || !anti[2]) return 5;

		long long ndiv[9][3][3]{};
		long long first_div[9][3][3];
		for (auto& aa : first_div) for (auto& nn : aa) for (long long& v : nn) v = -1;
		double pmin[3] = {1e300, 1e300, 1e300};
		double pmax[3] = {-1.0, -1.0, -1.0};
		std::vector<int64_t> qspreads[3];
		std::vector<int64_t> pomspreads[3][3];
		double cost_ns[3]{};
		{
			const int32_t* row = dump.logits.data();
			superslm::FsdTopK(row, mask.data(), vocab, 10, top10.data());
			constexpr int kRepetitions = 20000;
			for (int ki = 0; ki < 3; ++ki) {
				int64_t q[10]{};
				const auto begin = std::chrono::steady_clock::now();
				for (int rep = 0; rep < kRepetitions; ++rep) {
					if (!superslm::TopKRenormalizeQ15(row, top10.data(),
					                                      static_cast<size_t>(kWidths[ki]), kQLn2,
					                                      kQB, kQC, q)) return 10;
				}
				const auto elapsed = std::chrono::steady_clock::now() - begin;
				cost_ns[ki] = static_cast<double>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) /
				    kRepetitions;
				if (q[0] < 0) return 11;  // consume the output; q_theta is a probability.
			}
		}

		for (uint64_t r = 0; r < dump.rows; ++r) {
			const int32_t* row = dump.logits.data() + static_cast<size_t>(r * dump.vocab);
			superslm::FsdTopK(row, mask.data(), vocab, 10, top10.data());
			const int32_t greedy = top10[0];
			int64_t full_z = 0;
			if (!FullRowZ(row, static_cast<size_t>(dump.vocab), &wide, &shifted, &full_z)) {
				std::fprintf(stderr, "full-row i-exp refused: %s row %llu\n", name.c_str(),
				             static_cast<unsigned long long>(r));
				return 6;
			}

			for (int ki = 0; ki < 3; ++ki) {
				const int k = kWidths[ki];
				int64_t q[10]{};
				if (!superslm::TopKRenormalizeQ15(row, top10.data(), static_cast<size_t>(k),
				                                      kQLn2, kQB, kQC, q)) return 7;
				int64_t z_k = 0;
				const int64_t m = kQB * kQB + kQC;
				for (int j = 0; j < k; ++j) {
					int64_t value = 0;
					if (!IExpValue(shifted[static_cast<size_t>(top10[j])], m, &value)) return 8;
					z_k += value;
				}
				const double ptopk = static_cast<double>(z_k) / static_cast<double>(full_z);
				pmin[ki] = std::min(pmin[ki], ptopk);
				pmax[ki] = std::max(pmax[ki], ptopk);
				const auto [qlo, qhi] = std::minmax_element(q, q + k);
				qspreads[ki].push_back(*qhi - *qlo);

				for (int ni = 0; ni < 3; ++ni) {
					int64_t p[10]{};
					superslm::AntiLmPenalize(anti[ni], top10.data(), static_cast<size_t>(k), p);
					const auto [plo, phi] = std::minmax_element(p, p + k);
					pomspreads[ni][ki].push_back(*phi - *plo);
					for (int ai = 0; ai < 9; ++ai) {
						const int64_t alpha_q15 = static_cast<int64_t>(kAlphas[ai] * 32768.0 + 0.5);
						const int32_t selected = Select(top10.data(), q, p, k, alpha_q15);
						if (selected != greedy) {
							++ndiv[ai][ni][ki];
							if (first_div[ai][ni][ki] < 0) first_div[ai][ni][ki] = static_cast<long long>(r);
						}
					}
				}
			}

			// Commission the custom aggregation against the public production selector on every row.
			int32_t production_token = -1;
			bool refused = false;
			if (!superslm::DampedGreedyScoreAndArgmax(row, mask.data(), vocab, 10, anti[2], 0,
			                                              kQLn2, kQB, kQC, &production_token,
			                                              &refused) || refused || production_token != greedy) {
				std::fprintf(stderr, "production alpha=0 commissioning failed: %s row %llu\n",
				             name.c_str(), static_cast<unsigned long long>(r));
				return 9;
			}
			for (AntiLmState* state : anti) superslm::AntiLmUpdate(state, greedy);
		}

		for (int ki = 0; ki < 3; ++ki) {
			out << "C\t" << name << '\t' << kWidths[ki] << '\t' << cost_ns[ki]
			    << "\t20000\n";
			out << "S\t" << name << '\t' << kWidths[ki] << '\t' << dump.rows << '\t'
			    << pmin[ki] << '\t' << pmax[ki] << '\t'
			    << (pmin[ki] > 0 ? pmax[ki] / pmin[ki] : -1.0) << '\t'
			    << Median(qspreads[ki]) << '\n';
			for (int ni = 0; ni < 3; ++ni) {
				out << "SN\t" << name << '\t' << kOrders[ni] << '\t' << kWidths[ki] << '\t'
				    << dump.rows << '\t' << Median(pomspreads[ni][ki]) << '\n';
				for (int ai = 0; ai < 9; ++ai) {
					out << "A\t" << name << '\t' << kAlphas[ai] << '\t' << kOrders[ni] << '\t'
					    << kWidths[ki] << '\t' << dump.rows << '\t' << ndiv[ai][ni][ki]
					    << '\t' << first_div[ai][ni][ki] << '\n';
				}
			}
		}
		for (AntiLmState* state : anti) superslm::AntiLmDestroy(state);
		std::fprintf(stderr, "done %s (%llu rows)\n", name.c_str(),
		             static_cast<unsigned long long>(dump.rows));
	}
	return 0;
}
