// Our mesh test's optional amplification stage, with its own root block.
cbuffer Draw : register(b0) { float4 tint; };
struct Payload { float4 color; };
groupshared Payload data;
[RootSignature("RootConstants(num32BitConstants=4, b0)")]
[numthreads(1,1,1)]
void main() { data.color = tint; DispatchMesh(1,1,1,data); }
