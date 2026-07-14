#include "g1_skeleton.h"
#include <cassert>
#include <string_view>

int main() {
    using namespace g1_skeleton;
    static_assert(BoneCount == 31);
    static_assert(LeftToe == 7 && RightToe == 13);
    static_assert(LeftWrist == 23 && RightWrist == 30);
    assert(kParents[Simulation] == -1);
    assert(kParents[Hips] == Simulation);
    assert(kParents[LeftWrist] == LeftWristPitch);
    assert(kParents[RightWrist] == RightWristPitch);
    assert(kBoneNames[Spine2] == std::string_view("Spine2"));
    assert(kSkeletonSignature == std::string_view(
        "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7"));
}
