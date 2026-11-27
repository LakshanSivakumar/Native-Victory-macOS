// Synthetic execution probe: zero primitives, so a UAV proves the mesh stage ran
// independently of clipping, face culling, stage-in linkage and fragment output.
RWByteAddressBuffer observed : register(u0);
struct Vertex { float4 position : SV_Position; };
[RootSignature("UAV(u0)")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID,
          out vertices Vertex verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(0, 0);
    // The test dispatches a 2x2x2 grid. Give every lane of every group its own word.
    uint word = (group.x + 2 * group.y + 4 * group.z) * 32 + lane;
    observed.Store(word * 4, 0x4d530000u + word);
}
