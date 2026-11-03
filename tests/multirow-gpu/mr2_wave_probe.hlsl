// Multi-row GPU plan (rev 1.3.2) cell 4.9: a THROWAWAY probe (diagnostic only, never shipped, outside every
// widened-shader lint scope) that records the lane count the device runs at the widened sites' group shape
// ([numthreads(256, 1, 1)], plan E17/E21). Each thread writes WaveGetLaneCount() and its lane index; the host reads
// the distinct lane counts back.
RWByteAddressBuffer Out : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Out.Store(tid.x * 8, WaveGetLaneCount());
    Out.Store(tid.x * 8 + 4, WaveGetLaneIndex());
}
