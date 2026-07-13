"""
generate_database_g1.py  -- G1 variant of Holden's generate_database.py

Differences from the original (all marked #G1#):
  * files=[...] points at our exported G1 BVH (g1_takara_walk.bvh).
  * source is 50 fps (not LAFAN's 30); we resample to 60 fps to match the
    hard-coded `* 60.0` velocity scaling, and DROP the LAFAN 10% speed-up.
  * the LAFAN-only stats/visualisation block (reads simulation_run/walk.bin,
    hard-codes range indices 2 & 4, calls plt.show) is disabled.
  * sim_position_joint="Spine2", sim_rotation_joint="Hips", toes "LeftToe"/
    "RightToe" are UNCHANGED -- our BVH already uses those exact names.
"""
import quat
import bvh
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np

""" Basic function for mirroring animation data with this particular skeleton structure """

def animation_mirror(lrot, lpos, names, parents):

    joints_mirror = np.array([(
        names.index('Left'+n[5:]) if n.startswith('Right') else (
        names.index('Right'+n[4:]) if n.startswith('Left') else
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:,joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))

    return quat.ik(grot_mirror, gpos_mirror, parents)

""" Files to Process """

#G1# our single omnidirectional-walk clip (34863 frames @50fps). Trim 1 frame
#G1# off each end to avoid any edge artefacts; use the whole thing otherwise.
files = [
    ('g1_takara_walk.bvh', 1, 34862),
]

#G1# source / target frame rates
SRC_FPS = 50.0
DST_FPS = 60.0

""" We will accumulate data in these lists """

bone_positions = []
bone_velocities = []
bone_rotations = []
bone_angular_velocities = []
bone_parents = []
bone_names = []

range_starts = []
range_stops = []

contact_states = []

""" Loop Over Files """

for filename, start, stop in files:

    # For each file we process it mirrored and not mirrored
    for mirror in [False, True]:

        """ Load Data """

        print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))

        bvh_data = bvh.load(filename)
        bvh_data['positions'] = bvh_data['positions'][start:stop]
        bvh_data['rotations'] = bvh_data['rotations'][start:stop]

        positions = bvh_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

        # Convert from cm to m
        positions *= 0.01

        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
            rotations = quat.unroll(rotations)

        """ Supersample """

        nframes = positions.shape[0]
        nbones = positions.shape[1]

        #G1# resample SRC_FPS -> DST_FPS (no 10% speed-up)
        original_times = np.linspace(0, nframes - 1, nframes)
        n_out = int(round(nframes * DST_FPS / SRC_FPS))
        sample_times = np.linspace(0, nframes - 1, n_out)

        # This does a cubic interpolation of the data for supersampling
        positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])

        # Need to re-normalize after super-sampling
        rotations = quat.normalize(rotations)

        """ Extract Simulation Bone """

        # First compute world space positions/rotations
        global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])

        # Specify joints to use for simulation bone
        sim_position_joint = bvh_data['names'].index("Spine2")
        sim_rotation_joint = bvh_data['names'].index("Hips")

        # Position comes from spine joint
        sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
        sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')

        # Direction comes from projected hip forward direction
        #G1# LAFAN's Hips forward is local +Y; the G1 pelvis forward is local +X
        #G1# (toe geoms at +x, heel at -x). Local +Y is the robot's LEFT, so using
        #G1# +Y here would rotate every facing/trajectory feature ~90 deg. Measured:
        #G1# +X aligns with travel in 89% of moving frames, +Y in only 12%.
        sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([1.0, 0.0, 0.0]))

        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
        sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])

        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

        # Transform first joints to be local to sim and append sim as root bone
        positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
        rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])

        positions = np.concatenate([sim_position, positions], axis=1)
        rotations = np.concatenate([sim_rotation, rotations], axis=1)

        bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])

        bone_names = ['Simulation'] + bvh_data['names']

        """ Compute Velocities """

        # Compute velocities via central difference
        velocities = np.empty_like(positions)
        velocities[1:-1] = (
            0.5 * (positions[2:  ] - positions[1:-1]) * DST_FPS +
            0.5 * (positions[1:-1] - positions[ :-2]) * DST_FPS)
        velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

        # Same for angular velocities
        angular_velocities = np.zeros_like(positions)
        angular_velocities[1:-1] = (
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * DST_FPS +
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * DST_FPS)
        angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
        angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

        """ Compute Contact Data """

        global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
            rotations,
            positions,
            velocities,
            angular_velocities,
            bone_parents)

        contact_velocity_threshold = 0.15

        contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
            bone_names.index("LeftToe"),
            bone_names.index("RightToe")])]**2, axis=-1))

        # Contacts are given for when contact bones are below velocity threshold
        contacts = contact_velocity < contact_velocity_threshold

        # Median filter here acts as a kind of "majority vote", and removes
        # small regions  where contact is either active or inactive
        for ci in range(contacts.shape[1]):

            contacts[:,ci] = ndimage.median_filter(
                contacts[:,ci],
                size=6,
                mode='nearest')

        """ Append to Database """

        bone_positions.append(positions)
        bone_velocities.append(velocities)
        bone_rotations.append(rotations)
        bone_angular_velocities.append(angular_velocities)

        offset = 0 if len(range_starts) == 0 else range_stops[-1]

        range_starts.append(offset)
        range_stops.append(offset + len(positions))

        contact_states.append(contacts)


""" Concatenate Data """

bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
bone_parents = bone_parents.astype(np.int32)

range_starts = np.array(range_starts).astype(np.int32)
range_stops = np.array(range_stops).astype(np.int32)

contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

#G1# LAFAN-only stats/visualisation block disabled (needs simulation_run/walk.bin
#G1# and hard-coded range indices 2 & 4; also calls plt.show on a headless box).

""" Write Database """

print("Writing Database...")

with open('database.bin', 'wb') as f:

    nframes = bone_positions.shape[0]
    nbones = bone_positions.shape[1]
    nranges = range_starts.shape[0]
    ncontacts = contact_states.shape[1]

    f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_velocities.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_angular_velocities.ravel().tobytes())
    f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())

    f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
    f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())

    f.write(struct.pack('II', nframes, ncontacts) + contact_states.ravel().tobytes())

print("DONE: database.bin  nframes=%d nbones=%d nranges=%d ncontacts=%d" %
      (nframes, nbones, nranges, ncontacts))
