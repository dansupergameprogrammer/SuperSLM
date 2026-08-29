// T-2380 (Curie) -- the forty-seventh/forty-eighth dimension-11 populations'
// own floating-point carrier (design Sec7 dim 11, D-SLM5053/D-SLM5068),
// adopted UNMODIFIED from the adversary's own construction: T-2376's
// Claude/Loki/t2376-probe/fp_carrier.cpp, copied here verbatim rather than
// re-derived (StandardsDocument.md Sec5.4's commissioning rule: a must-reject
// construction is authored by a seat independent of the instrument's maker,
// and re-deriving it here would forfeit that independence).
//
// One extern "C" function whose body is genuine IEEE-754 floating-point
// arithmetic (multiply + add, divide) on doubles. Compiled to a real COFF
// object and archived into a real `!<arch>` container with lib.exe, this is
// the one object member every dimension-11 population in this file's
// companion test_archive_composition.py inserts at a chosen archive position
// to construct a must-reject the honest driver blocks and a membership-
// dropping driver silently ships.
extern "C" double SslmProbeFpCarrier(double a, double b) {
    return a * b + a / b;
}
