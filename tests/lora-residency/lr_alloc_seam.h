// Streamed adapters (plan: superslm-lora-residency rev 1.2, §8): the allocation seam. A process-wide
// replacement of every global operator new and delete (lr_alloc_seam.cpp, linked only into the
// lora-residency suite) that counts the requested bytes of each allocation made on a thread while
// that thread is armed, and keeps a high-water mark of the live counted bytes. A counted block is
// marked in its own header, so freeing it on any thread lowers the live count.
//
// The cells read: Live() (counted bytes not yet freed), Hwm() (the most Live() has been since the
// last ResetHwm), Count() (counted allocations). Never the engine's own figures.
#ifndef SUPERSLM_TESTS_LORA_RESIDENCY_LR_ALLOC_SEAM_H
#define SUPERSLM_TESTS_LORA_RESIDENCY_LR_ALLOC_SEAM_H

#include <cstdint>

namespace lr_seam {

void Arm();     // count this thread's allocations from now on
void Disarm();  // stop counting this thread's allocations (frees of counted blocks still count)
int64_t Live();
int64_t Hwm();
int64_t Count();
void ResetHwm();  // Hwm() = Live()

// Arms the calling thread for its scope and reports what the scope allocated: the high-water mark
// above the live count at entry, and the net live bytes left at exit.
class Scope {
public:
	Scope();
	~Scope();
	int64_t PeakAboveEntry() const;  // Hwm() - live at entry
	int64_t NetLive() const;         // Live() - live at entry
	int64_t Allocations() const;     // Count() - count at entry

private:
	int64_t live0_, count0_;
	bool was_armed_;
};

}  // namespace lr_seam

#endif  // SUPERSLM_TESTS_LORA_RESIDENCY_LR_ALLOC_SEAM_H
