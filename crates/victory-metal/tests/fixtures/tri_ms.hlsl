// Our offscreen mesh test. No game code or data.
cbuffer Draw : register(b0) { float4 tint; };
struct Vertex { float4 position : SV_Position; float4 color : TEXCOORD0; };
struct Payload { float4 color; };
#ifndef AMPLIFIED
[RootSignature("RootConstants(num32BitConstants=4, b0)")]
#endif
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID,
#ifdef AMPLIFIED
          in payload Payload data,
#endif
          out vertices Vertex verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    if (lane < 3) {
        float2 positions[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
        verts[lane].position = float4(positions[lane], 0.5, 1);
#ifdef AMPLIFIED
        verts[lane].color = data.color;
#else
        verts[lane].color = tint;
#endif
    }
    if (lane == 0) tris[0] = uint3(0,1,2);
}
