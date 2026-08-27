// T-2326 (Curie) -- Sec7 dimension 11's sixth commissioning population: a
// translation unit added to the build's compiled source set but NOT mirrored
// into whatever list the scanner's own TU enumeration reads from (design
// Sec4.1's own text: the production instrument's TU enumeration "derives...
// from SUPERSLM_CORE_SOURCES (CMakeLists.txt:30-47)... at scan time... rather
// than maintaining a second, hand-written list a future translation unit
// could be added without").
//
// CONSTRUCTION. pop06_mini_source_list.txt (this directory) names TWO files
// as "the build's own compiled source set" -- pop06_tu_a.cpp and this file --
// mirroring CMakeLists.txt's own SUPERSLM_CORE_SOURCES shape (a flat list a
// real build compiles every entry of). pop06_mini_scan_list_stale.txt names
// only pop06_tu_a.cpp -- a hand-maintained list that agreed with the source
// set on the day it was written and was never updated when this file was
// added, exactly the drift D-SLM4335 names. This file's own CalleeInOtherTU
// is called from pop06_tu_a.cpp's Root (one hop) and contains a flagged
// floating-point instruction. A scan whose TU enumeration is DERIVED from the
// source-set file reports the flagged instruction; a scan whose TU
// enumeration reads the stale hand-maintained list never opens this object at
// all and reports clean -- the identical shape T-2270's own fracture found
// one level up (a root the walk never reached), here at the level of "was
// this file's own object ever handed to the scanner in the first place."
//
// Build: clang, either ELF or COFF target -- no STL, no toolchain constraint.

extern "C" __attribute__((noinline)) double CalleeInOtherTU(double x, double y) {
    return x / y;  // the flagged instruction (a genuine divsd), living ONLY here
}
