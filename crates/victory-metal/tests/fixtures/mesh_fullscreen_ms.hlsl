// Synthetic full-screen CCW triangle: no resources, root reads, or user varyings.
struct Vertex { float4 position : SV_Position; };
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(uint lane : SV_GroupIndex,
          out vertices Vertex verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    if (lane < 3) {
        float2 positions[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
        verts[lane].position = float4(positions[lane], 0.5, 1);
    }
    if (lane == 0) tris[0] = uint3(0,1,2);
}
