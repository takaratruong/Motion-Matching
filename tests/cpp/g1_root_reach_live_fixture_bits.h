#ifndef G1_ROOT_REACH_LIVE_FIXTURE_BITS_H
#define G1_ROOT_REACH_LIVE_FIXTURE_BITS_H

#include <cstdint>

struct G1TestRootReachTargetBits
{
    std::uint8_t flags[4];
    std::uint32_t scalars[13];
};

struct G1TestRootReachLiveFixtureBits
{
    const std::uint32_t (*position_bits)[3];
    const std::uint32_t (*rotation_bits)[4];
    const std::int32_t* parents;
    const std::uint8_t* contacts;
    const G1TestRootReachTargetBits* targets;
};

static constexpr std::uint32_t G1TestRootReachRow6PositionBits[31][3] = {
    {UINT32_C(0xbde416e1), UINT32_C(0x00000000), UINT32_C(0xbf06dde4)},
    {UINT32_C(0x3acb2ca7), UINT32_C(0x3f48ed9e), UINT32_C(0x3bb62ecf)},
    {UINT32_C(0xa532cad6), UINT32_C(0xbdd25461), UINT32_C(0xbd83ff69)},
    {UINT32_C(0x24680cbb), UINT32_C(0xbcf991bc), UINT32_C(0xbd54fdf4)},
    {UINT32_C(0x3ccccee6), UINT32_C(0xbdfe32a0), UINT32_C(0xa487e4f7)},
    {UINT32_C(0xbda04d98), UINT32_C(0xbe35989e), UINT32_C(0xbb0cd48f)},
    {UINT32_C(0x251c3576), UINT32_C(0xbe999ae9), UINT32_C(0x38c610c6)},
    {UINT32_C(0xa519d0d1), UINT32_C(0xbc8fd5cb), UINT32_C(0xa43fb41a)},
    {UINT32_C(0xa4601c10), UINT32_C(0xbdd25461), UINT32_C(0x3d83ff69)},
    {UINT32_C(0x2504d783), UINT32_C(0xbcf991bc), UINT32_C(0x3d54fdf4)},
    {UINT32_C(0x3ccccee6), UINT32_C(0xbdfe32a0), UINT32_C(0x235d2dc8)},
    {UINT32_C(0xbda04d98), UINT32_C(0xbe35989e), UINT32_C(0x3b0cd48f)},
    {UINT32_C(0x240feeec), UINT32_C(0xbe999ae9), UINT32_C(0xb8c610c6)},
    {UINT32_C(0x24acb206), UINT32_C(0xbc8fd5cb), UINT32_C(0x2432e5cb)},
    {UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000)},
    {UINT32_C(0xbb81e03f), UINT32_C(0x3d0f5c29), UINT32_C(0xa41db09d)},
    {UINT32_C(0xa555326b), UINT32_C(0x3c9ba5e3), UINT32_C(0xa314c8f0)},
    {UINT32_C(0x3b81a3da), UINT32_C(0x3e737c9a), UINT32_C(0xbdcd4025)},
    {UINT32_C(0x251bc688), UINT32_C(0xbc629b6b), UINT32_C(0xbd1ba5e3)},
    {UINT32_C(0x2589bae0), UINT32_C(0xbdd35a86), UINT32_C(0xbbcc78ea)},
    {UINT32_C(0x3c814b5a), UINT32_C(0xbda4e69f), UINT32_C(0xa4a9b8e9)},
    {UINT32_C(0x3dcccccd), UINT32_C(0xbc23d70a), UINT32_C(0xbaf773bf)},
    {UINT32_C(0x3d1ba5e3), UINT32_C(0x24ae7cb6), UINT32_C(0xa5187d55)},
    {UINT32_C(0x3d3c6a7f), UINT32_C(0x24a6de78), UINT32_C(0xa43d0e38)},
    {UINT32_C(0x3b81a3da), UINT32_C(0x3e737c9a), UINT32_C(0x3dcd3ae7)},
    {UINT32_C(0xa59e7370), UINT32_C(0xbc629b6b), UINT32_C(0x3d1ba5e3)},
    {UINT32_C(0x252384b5), UINT32_C(0xbdd35a86), UINT32_C(0x3bcc78ea)},
    {UINT32_C(0x3c814b5a), UINT32_C(0xbda4e69f), UINT32_C(0x248958fb)},
    {UINT32_C(0x3dcccccd), UINT32_C(0xbc23d70a), UINT32_C(0x3af773bf)},
    {UINT32_C(0x3d1ba5e3), UINT32_C(0xa3bf5ac0), UINT32_C(0xa0fef500)},
    {UINT32_C(0x3d3c6a7f), UINT32_C(0xa29a9d50), UINT32_C(0x252fba99)},
};

static constexpr std::uint32_t G1TestRootReachRow6RotationBits[31][4] = {
    {UINT32_C(0x3f7e613e), UINT32_C(0x00000000), UINT32_C(0x3de60c3c), UINT32_C(0x80000000)},
    {UINT32_C(0x3f35c298), UINT32_C(0xba484330), UINT32_C(0xbf343859), UINT32_C(0x3c8ed7b3)},
    {UINT32_C(0x3f7ffd9d), UINT32_C(0x22a5191d), UINT32_C(0xa5359b1f), UINT32_C(0x3c0bc3d4)},
    {UINT32_C(0x3f7eed7d), UINT32_C(0x3cdd9ce8), UINT32_C(0x3b1b6fd4), UINT32_C(0x3db2cdad)},
    {UINT32_C(0x3f7f77c5), UINT32_C(0x25079437), UINT32_C(0x3d83fb55), UINT32_C(0xa4e5cd5d)},
    {UINT32_C(0x3f78fc85), UINT32_C(0xa500344b), UINT32_C(0xa5da42e5), UINT32_C(0xbe6e0cff)},
    {UINT32_C(0x3f7ebc16), UINT32_C(0xa3db647e), UINT32_C(0x25708d3d), UINT32_C(0x3dcb5e3e)},
    {UINT32_C(0x3f7fadfe), UINT32_C(0xbd4cd86e), UINT32_C(0x24241bc7), UINT32_C(0x23834510)},
    {UINT32_C(0x3f7f9fc3), UINT32_C(0xa50498f6), UINT32_C(0x2540bd05), UINT32_C(0x3d5de528)},
    {UINT32_C(0x3f7eda0d), UINT32_C(0xbd14f45f), UINT32_C(0xbb50f33e), UINT32_C(0x3db2c00c)},
    {UINT32_C(0x3f7fdc8b), UINT32_C(0xa4453739), UINT32_C(0xbd06b5f8), UINT32_C(0x228b506e)},
    {UINT32_C(0x3f755cb5), UINT32_C(0x222f767c), UINT32_C(0x2477f0f2), UINT32_C(0xbe920f7b)},
    {UINT32_C(0x3f7df6c8), UINT32_C(0x24d19fa3), UINT32_C(0xa4c50b89), UINT32_C(0x3e00e3d4)},
    {UINT32_C(0x3f7fff46), UINT32_C(0x3b99af68), UINT32_C(0xa55d8506), UINT32_C(0xa4ea6c01)},
    {UINT32_C(0x3f7feb7a), UINT32_C(0xa41f14f6), UINT32_C(0xbccd00ed), UINT32_C(0xa39823d6)},
    {UINT32_C(0x3f7febed), UINT32_C(0xbccabbd0), UINT32_C(0xa41d2b25), UINT32_C(0xa422df42)},
    {UINT32_C(0x3f7fb984), UINT32_C(0xa5330cd1), UINT32_C(0x2560529a), UINT32_C(0x3d3de99c)},
    {UINT32_C(0x3f7a9fe6), UINT32_C(0x3e0ce7c2), UINT32_C(0xbcac4dd5), UINT32_C(0x3e187ea5)},
    {UINT32_C(0x3f7e489c), UINT32_C(0x3decc17a), UINT32_C(0xa3ac3e44), UINT32_C(0x24bafd41)},
    {UINT32_C(0x3f7d6612), UINT32_C(0xa4ef6426), UINT32_C(0xbe119b75), UINT32_C(0xa48208fd)},
    {UINT32_C(0x3f58b6a1), UINT32_C(0xa40c1a16), UINT32_C(0x23a42a5c), UINT32_C(0xbf0846cc)},
    {UINT32_C(0x3f7ee74a), UINT32_C(0x3dbd5a3b), UINT32_C(0x24cacd61), UINT32_C(0x255afea2)},
    {UINT32_C(0x3f7ffde6), UINT32_C(0xa57cb3e3), UINT32_C(0xa4460216), UINT32_C(0x3c030bd4)},
    {UINT32_C(0x3f7ffc3c), UINT32_C(0x25e4c0dc), UINT32_C(0x3c2f8dc3), UINT32_C(0xa5be5431)},
    {UINT32_C(0x3f7d4ff6), UINT32_C(0xbe0e6d92), UINT32_C(0x3bb649f7), UINT32_C(0x3d1f2b12)},
    {UINT32_C(0x3f7d31c7), UINT32_C(0xbe172f9b), UINT32_C(0xa55611c6), UINT32_C(0x2491e733)},
    {UINT32_C(0x3f7e0d12), UINT32_C(0xa3a3d1cd), UINT32_C(0x3dfc3ac0), UINT32_C(0xa4ba1020)},
    {UINT32_C(0x3f618b37), UINT32_C(0x2309a3a4), UINT32_C(0xa527e360), UINT32_C(0xbef234d9)},
    {UINT32_C(0x3f7fe5b0), UINT32_C(0xbce819f2), UINT32_C(0xa502c61c), UINT32_C(0xa4a013a1)},
    {UINT32_C(0x3f7f989b), UINT32_C(0xa4420930), UINT32_C(0xa20ea928), UINT32_C(0xbd65fd94)},
    {UINT32_C(0x3f7f2cdc), UINT32_C(0xa4a77061), UINT32_C(0xbda44315), UINT32_C(0x23d53a02)},
};

static constexpr std::int32_t G1TestRootReachRow6Parents[31] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
};

static constexpr std::uint8_t G1TestRootReachRow6Contacts[2] = {0, 1};

static constexpr G1TestRootReachTargetBits G1TestRootReachRow6Targets[2] = {
    {{0, 1, 1, 0}, {UINT32_C(0x3cc9d358), UINT32_C(0x3ba3d70a), UINT32_C(0xbf13e07b), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0x3d83f15c), UINT32_C(0x3f7f6813), UINT32_C(0x3cb38c12), UINT32_C(0x3d7ab6bb), UINT32_C(0x3bce206f), UINT32_C(0xbf1a8b3a), UINT32_C(0x00000000)}},
    {{1, 1, 0, 0}, {UINT32_C(0xbe8049ae), UINT32_C(0x3ba3d70a), UINT32_C(0xbf03d7e0), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0xbe8049ae), UINT32_C(0x3ba3d70a), UINT32_C(0xbf03d7e0), UINT32_C(0x3cf67c19)}},
};

static constexpr G1TestRootReachLiveFixtureBits G1TestRootReachRow6 = {
    G1TestRootReachRow6PositionBits, G1TestRootReachRow6RotationBits,
    G1TestRootReachRow6Parents, G1TestRootReachRow6Contacts, G1TestRootReachRow6Targets
};

static constexpr std::uint32_t G1TestRootReachRow7PositionBits[31][3] = {
    {UINT32_C(0xbdddb1d1), UINT32_C(0x00000000), UINT32_C(0xbf07a831)},
    {UINT32_C(0x3aad8ed8), UINT32_C(0x3f49703b), UINT32_C(0x3baa0ab9)},
    {UINT32_C(0xa45ddb5c), UINT32_C(0xbdd25461), UINT32_C(0xbd83ff69)},
    {UINT32_C(0x244ea6da), UINT32_C(0xbcf991bc), UINT32_C(0xbd54fdf4)},
    {UINT32_C(0x3ccccee6), UINT32_C(0xbdfe32a0), UINT32_C(0xa3234fac)},
    {UINT32_C(0xbda04d98), UINT32_C(0xbe35989e), UINT32_C(0xbb0cd48f)},
    {UINT32_C(0x2511e882), UINT32_C(0xbe999ae9), UINT32_C(0x38c610c6)},
    {UINT32_C(0x248651ab), UINT32_C(0xbc8fd5cb), UINT32_C(0x24086f61)},
    {UINT32_C(0xa396f460), UINT32_C(0xbdd25461), UINT32_C(0x3d83ff69)},
    {UINT32_C(0x250dfe92), UINT32_C(0xbcf991bc), UINT32_C(0x3d54fdf4)},
    {UINT32_C(0x3ccccee6), UINT32_C(0xbdfe32a0), UINT32_C(0xa3004df0)},
    {UINT32_C(0xbda04d98), UINT32_C(0xbe35989e), UINT32_C(0x3b0cd48f)},
    {UINT32_C(0x220de307), UINT32_C(0xbe999ae9), UINT32_C(0xb8c610c6)},
    {UINT32_C(0x245bcc3a), UINT32_C(0xbc8fd5cb), UINT32_C(0xa3b20786)},
    {UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000)},
    {UINT32_C(0xbb81e03f), UINT32_C(0x3d0f5c29), UINT32_C(0xa50b53d6)},
    {UINT32_C(0xa3995038), UINT32_C(0x3c9ba5e3), UINT32_C(0x246d294e)},
    {UINT32_C(0x3b81a3da), UINT32_C(0x3e737c9a), UINT32_C(0xbdcd4025)},
    {UINT32_C(0x252dc494), UINT32_C(0xbc629b6b), UINT32_C(0xbd1ba5e3)},
    {UINT32_C(0x24f3e721), UINT32_C(0xbdd35a86), UINT32_C(0xbbcc78ea)},
    {UINT32_C(0x3c814b5a), UINT32_C(0xbda4e69f), UINT32_C(0xa500033f)},
    {UINT32_C(0x3dcccccd), UINT32_C(0xbc23d70a), UINT32_C(0xbaf773bf)},
    {UINT32_C(0x3d1ba5e3), UINT32_C(0x250377db), UINT32_C(0xa530b869)},
    {UINT32_C(0x3d3c6a7f), UINT32_C(0x2532cc47), UINT32_C(0xa42d21fc)},
    {UINT32_C(0x3b81a3da), UINT32_C(0x3e737c9a), UINT32_C(0x3dcd3ae7)},
    {UINT32_C(0xa4cab4c0), UINT32_C(0xbc629b6b), UINT32_C(0x3d1ba5e3)},
    {UINT32_C(0x24e0528f), UINT32_C(0xbdd35a86), UINT32_C(0x3bcc78ea)},
    {UINT32_C(0x3c814b5a), UINT32_C(0xbda4e69f), UINT32_C(0xa44b34e7)},
    {UINT32_C(0x3dcccccd), UINT32_C(0xbc23d70a), UINT32_C(0x3af773bf)},
    {UINT32_C(0x3d1ba5e3), UINT32_C(0xa1f49780), UINT32_C(0x22860810)},
    {UINT32_C(0x3d3c6a7f), UINT32_C(0xa3d5217c), UINT32_C(0x23c05258)},
};

static constexpr std::uint32_t G1TestRootReachRow7RotationBits[31][4] = {
    {UINT32_C(0x3f7e54dc), UINT32_C(0x00000000), UINT32_C(0x3de971a8), UINT32_C(0x80000000)},
    {UINT32_C(0x3f357d67), UINT32_C(0xbb298308), UINT32_C(0xbf34886b), UINT32_C(0x3c0d541c)},
    {UINT32_C(0x3f7ffe4d), UINT32_C(0x252e31b0), UINT32_C(0x2488ad8a), UINT32_C(0x3bebf8c1)},
    {UINT32_C(0x3f7eef66), UINT32_C(0x3cd4a2f0), UINT32_C(0x3b152414), UINT32_C(0x3db2cf05)},
    {UINT32_C(0x3f7f882f), UINT32_C(0xa48ecece), UINT32_C(0x3d77907e), UINT32_C(0x248c855f)},
    {UINT32_C(0x3f7a4d97), UINT32_C(0xa3b2e620), UINT32_C(0xa1a9368c), UINT32_C(0xbe56d354)},
    {UINT32_C(0x3f7ef123), UINT32_C(0x23567305), UINT32_C(0xa5a1290b), UINT32_C(0x3dba01cd)},
    {UINT32_C(0x3f7fd7f5), UINT32_C(0xbd0f294b), UINT32_C(0xa4722125), UINT32_C(0x24abe2d4)},
    {UINT32_C(0x3f7fb5be), UINT32_C(0x24949fbd), UINT32_C(0xa4a0b582), UINT32_C(0x3d42ee46)},
    {UINT32_C(0x3f7ed763), UINT32_C(0xbd197201), UINT32_C(0xbb573fec), UINT32_C(0x3db2be2e)},
    {UINT32_C(0x3f7fc0a1), UINT32_C(0xa41c6b58), UINT32_C(0xbd341490), UINT32_C(0x25068085)},
    {UINT32_C(0x3f772d80), UINT32_C(0xa2b17d84), UINT32_C(0x24dda97a), UINT32_C(0xbe854108)},
    {UINT32_C(0x3f7e7020), UINT32_C(0xa4b179ef), UINT32_C(0x249258c4), UINT32_C(0x3de1e4ad)},
    {UINT32_C(0x3f7ffc4e), UINT32_C(0x3c2e020c), UINT32_C(0x2326c6e5), UINT32_C(0xa44be2c9)},
    {UINT32_C(0x3f7fef4b), UINT32_C(0x24c74e04), UINT32_C(0xbcb8f7ca), UINT32_C(0xa501bafe)},
    {UINT32_C(0x3f7ff553), UINT32_C(0xbc93d892), UINT32_C(0x2495abf9), UINT32_C(0xa4935da0)},
    {UINT32_C(0x3f7fbf43), UINT32_C(0xa4d874a9), UINT32_C(0xa4364052), UINT32_C(0x3d3603fe)},
    {UINT32_C(0x3f7a878a), UINT32_C(0x3e0cd9fe), UINT32_C(0xbcaf182d), UINT32_C(0x3e1af9ed)},
    {UINT32_C(0x3f7e5e0b), UINT32_C(0x3de6edb8), UINT32_C(0xa5822c88), UINT32_C(0x2454d582)},
    {UINT32_C(0x3f7d67af), UINT32_C(0xa46caecc), UINT32_C(0xbe116e91), UINT32_C(0xa4a53fc4)},
    {UINT32_C(0x3f5833b2), UINT32_C(0xa560fa91), UINT32_C(0x24ab563e), UINT32_C(0xbf091628)},
    {UINT32_C(0x3f7eee92), UINT32_C(0x3dbae28d), UINT32_C(0xa59eb1e6), UINT32_C(0x256870a0)},
    {UINT32_C(0x3f7ffe2d), UINT32_C(0x24a990b7), UINT32_C(0x258802c8), UINT32_C(0x3bf47010)},
    {UINT32_C(0x3f7ffbc9), UINT32_C(0x24df1931), UINT32_C(0x3c39c33a), UINT32_C(0xa57691ea)},
    {UINT32_C(0x3f7d3cae), UINT32_C(0xbe0e628c), UINT32_C(0x3bd5fd5a), UINT32_C(0x3d3b5bdf)},
    {UINT32_C(0x3f7d5712), UINT32_C(0xbe133acf), UINT32_C(0xa5589a7d), UINT32_C(0x24bd2312)},
    {UINT32_C(0x3f7e4cb0), UINT32_C(0xa09821e0), UINT32_C(0x3deba8b0), UINT32_C(0x2410dd9d)},
    {UINT32_C(0x3f6072ce), UINT32_C(0xa51f53a0), UINT32_C(0xa3889b08), UINT32_C(0xbef63e31)},
    {UINT32_C(0x3f7fe47b), UINT32_C(0xbced6265), UINT32_C(0xa51590ca), UINT32_C(0x2476ecf2)},
    {UINT32_C(0x3f7f90af), UINT32_C(0xa57caf9b), UINT32_C(0x24580a9c), UINT32_C(0xbd6ea262)},
    {UINT32_C(0x3f7f2af9), UINT32_C(0x252a39e7), UINT32_C(0xbda4fe5e), UINT32_C(0xa4a464ac)},
};

static constexpr std::int32_t G1TestRootReachRow7Parents[31] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
};

static constexpr std::uint8_t G1TestRootReachRow7Contacts[2] = {1, 1};

static constexpr G1TestRootReachTargetBits G1TestRootReachRow7Targets[2] = {
    {{1, 1, 0, 0}, {UINT32_C(0x3d7ac22f), UINT32_C(0x3ba3d70a), UINT32_C(0xbf1a8352), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0x3d7ac22f), UINT32_C(0x3ba3d70a), UINT32_C(0xbf1a8352), UINT32_C(0x3d4e0c8c)}},
    {{1, 1, 0, 0}, {UINT32_C(0xbe8049ae), UINT32_C(0x3ba3d70a), UINT32_C(0xbf03d7e0), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3f800000), UINT32_C(0x00000000), UINT32_C(0xbe8049ae), UINT32_C(0x3ba3d70a), UINT32_C(0xbf03d7e0), UINT32_C(0x3cd4888d)}},
};

static constexpr G1TestRootReachLiveFixtureBits G1TestRootReachRow7 = {
    G1TestRootReachRow7PositionBits, G1TestRootReachRow7RotationBits,
    G1TestRootReachRow7Parents, G1TestRootReachRow7Contacts, G1TestRootReachRow7Targets
};

#endif
