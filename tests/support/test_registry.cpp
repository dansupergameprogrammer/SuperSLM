#include "test_registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace superslm_test_registry {

namespace {

struct Entry {
	std::string name;
	TestFn fn;
	int order;
};

std::vector<Entry>& Registry() {
	static std::vector<Entry> registry;
	return registry;
}

std::vector<Entry> SortedByOrder() {
	std::vector<Entry> sorted = Registry();
	std::sort(sorted.begin(), sorted.end(),
	          [](const Entry& a, const Entry& b) { return a.order < b.order; });
	return sorted;
}

}  // namespace

void RegisterTest(const char* name, TestFn fn, int order) {
	auto& reg = Registry();
	for (const auto& e : reg) {
		if (e.name == name) {
			std::fprintf(stderr,
			             "RegisterTest: duplicate test name '%s' -- already registered at "
			             "order %d\n",
			             name, e.order);
			std::fflush(stderr);
			std::abort();
		}
		if (e.order == order) {
			std::fprintf(stderr,
			             "RegisterTest: duplicate order key %d -- '%s' already claims it, "
			             "'%s' collides\n",
			             order, e.name.c_str(), name);
			std::fflush(stderr);
			std::abort();
		}
	}
	reg.push_back(Entry{name, fn, order});
}

size_t RegisteredTestCount() { return Registry().size(); }

void ListRegisteredTests() {
	for (const auto& e : SortedByOrder()) {
		std::printf("%s,%d\n", e.name.c_str(), e.order);
	}
}

void RunAllRegisteredTests() {
	std::vector<Entry> sorted = SortedByOrder();
	for (size_t i = 0; i < sorted.size(); ++i) {
		if (sorted[i].order != static_cast<int>(i)) {
			std::fprintf(stderr,
			             "RunAllRegisteredTests: registered order keys are not a dense "
			             "permutation of [0, %zu) -- expected order key %zu at sorted "
			             "position %zu, found %d ('%s') instead\n",
			             sorted.size(), i, i, sorted[i].order, sorted[i].name.c_str());
			std::fflush(stderr);
			std::abort();
		}
	}
	for (const auto& e : sorted) {
		e.fn();
	}
}

}  // namespace superslm_test_registry
