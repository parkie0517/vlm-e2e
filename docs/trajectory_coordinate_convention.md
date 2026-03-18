# Trajectory Coordinate Convention

## Problem

The ground-truth ego future trajectory initially projected incorrectly onto
`CAM_FRONT`. The camera projection code itself was correct, but the trajectory
coordinates were not in the ego-frame convention expected by the camera
calibration.

## Root Cause

The helper
`nuscenes.prediction.convert_global_coords_to_local()`
does not return coordinates in the standard nuScenes ego frame used by camera
extrinsics.

Its output should be interpreted as:

- `[:, 0] = lateral`, positive to the right
- `[:, 1] = longitudinal`, positive forward

But for this project, downstream camera projection and trajectory consumers use
the standard ego-frame convention:

- `x = forward`
- `y = left`

Without remapping, the trajectory points are fed into camera projection with the
wrong axes and end up off-screen or visually incorrect.

## Correct Convention

After conversion from global coordinates, the project remaps the trajectory into
the ego frame:

```python
ego = np.column_stack([local[:, 1], -local[:, 0]])
```

This means:

- `ego[:, 0]` is forward distance
- `ego[:, 1]` is leftward distance

## Where The Fix Is Applied

The remap is applied in:

- `dataset/drivelm_uniad_dataset.py`

Specifically in both:

- `_build_ego_future()`
- `_build_ego_history()`

## Downstream Rule

All downstream code should treat:

- `ego_past_traj[:, 0]` and `ego_future_traj[:, 0]` as forward
- `ego_past_traj[:, 1]` and `ego_future_traj[:, 1]` as left

Visualization, loss computation, and future model heads should use this
corrected ego-frame convention directly and should not apply any extra axis
swaps unless explicitly justified.
