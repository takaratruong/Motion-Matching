# PFNN final-pose export (`PFNNXFM/v1`)

Run from this directory so the released assets resolve:

```bash
./pfnn_export --export /tmp/pfnn-frames.bin
./pfnn_export --export /tmp/pfnn-frames.fifo
./pfnn_export --export /tmp/eight-frames.bin --export-frames 8
```

`--export` is opt-in. Without it, the published runtime and keyboard behavior
are unchanged. A FIFO must have a reader before startup because `fopen(...,
"wb")` intentionally blocks until the consumer attaches. Every frame is
flushed after it is written.

All integers and IEEE-754 floats are little-endian. Coordinates are Daniel
Holden's Y-up world in the released demo's centimeter-scale units. Quaternion
order is WXYZ. Each pose is captured after the released terrain IK and the
final forward-kinematics pass.

## 32-byte stream header

| Offset | Type | Value |
|---:|---|---|
| 0 | `char[8]` | `PFNNXFM\0` |
| 8 | `uint32` | version `1` |
| 12 | `uint32` | joint count `31` |
| 16 | `uint32` | record bytes `912` |
| 20 | `uint32` | coordinate code `1` = Holden Y-up centimeters |
| 24 | `float32` | FPS `60` |
| 28 | `uint32` | reserved `0` |

Python format: `struct.Struct("<8sIIIIfI")`.

## 912-byte frame record

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `uint64` | zero-based frame counter |
| 8 | `uint32` | active published world/heightmap ID, `0..5` |
| 12 | `uint32` | flags; bit 0 = post-reference-IK/final-FK |
| 16 | `float32` | gait phase in radians |
| 20 | `float32` | terrain height sampled below trajectory root, cm |
| 24 | `float32[3]` | trajectory root XYZ, cm |
| 36 | `float32[2]` | trajectory forward XZ |
| 44 | `float32[31][7]` | global joint `px,py,pz,qw,qx,qy,qz` |

Frame-prefix Python format: `struct.Struct("<QIIff3f2f")`; the following joint
payload is `struct.Struct("<" + "f" * (31 * 7))`.

Joint order is exactly:

1. Hips
2. LHipJoint
3. LeftUpLeg
4. LeftLeg
5. LeftFoot
6. LeftToeBase
7. RHipJoint
8. RightUpLeg
9. RightLeg
10. RightFoot
11. RightToeBase
12. LowerBack
13. Spine
14. Spine2 (released source name `Spine1`)
15. Neck
16. Neck1
17. Head
18. LeftShoulder
19. LeftArm
20. LeftForeArm
21. LeftHand
22. LeftFingerBase
23. LeftHandIndex1
24. LThumb
25. RightShoulder
26. RightArm
27. RightForeArm
28. RightHand
29. RightFingerBase
30. RightHandIndex1
31. RThumb
