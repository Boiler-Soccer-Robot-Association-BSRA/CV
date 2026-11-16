# Floor grid for the projection accuracy check

What to tape on the floor, what to measure and which files to fill in, for
the day the camera is calibrated and on the tripod. It feeds
[`../projection_accuracy.py`](../projection_accuracy.py), which checks the
milestone: position error ≤ 0.3 m at 2 m.

Allow about an hour. You need: the calibrated camera on the tripod, masking
tape, a tape measure (3 m or longer), a marker, a string with a weight (or
a phone level) and, for part B, the 10 cm ball.

## The layout: 13 points

`+x` is straight ahead of the camera and `+y` is to its **left**, both in
meters from the origin `O`, the spot on the floor directly under the lens.

```
        y = +1.0   +0.5     0     -0.5   -1.0
        (left)                           (right)

x = 2.5                   C250
x = 2.0   LL200   L200    C200    R200   RR200     <- the milestone row
x = 1.5           L150    C150    R150
x = 1.0           L100    C100    R100
x = 0.5                   C050

x = 0                       O    (under the lens; the camera looks up this page)
```

Five of the points are within 0.25 m of 2 m range, which is what the
milestone check needs. The others show how the error changes with distance
and off to the sides.

## Setting up

1. **Camera.** Set the tripod to the robot's camera height and tilt. Use
   the resolution the camera was calibrated at (1920x1200 unless that
   changed). Don't touch the lens.
2. **Origin.** Hang the weighted string from the lens and mark where it
   touches the floor. That is `O`.
3. **Centerline.** Run a strip of tape from `O` straight ahead, along the
   direction the camera faces, 2.5 m long. To get the direction, look at
   the live image and put the far end of the tape in the middle column of
   the picture.
4. **Points.** Mark 0.5, 1.0, 1.5, 2.0 and 2.5 m along the centerline. At
   1.0, 1.5 and 2.0 m, measure 0.5 m to each side at a right angle (a sheet
   of paper's corner is square enough); at 2.0 m also 1.0 m to each side.
   Make each mark a small tape cross. The **center of the cross** is the
   point.
5. **Check two diagonals** with the tape measure: `O` to `L200` and `O` to
   `R200` should both be 2.06 m. If they differ by more than 2 cm, the side
   points aren't square to the centerline; fix them before going on.
6. **Measure the mount** and write it into the `mount:` section of
   `camera_params.yaml` (`CALIBRATION_GUIDE.md`, Step 6): lens height above
   the floor, and the downward tilt in degrees. These two numbers matter
   most: 1 degree and 2 cm off moves a point at 2 m by about 6 cm.

If a mark isn't in the picture (`C050` is the likely one, at the bottom
edge), delete its row from the CSV rather than guessing.

## Part A: pixel mode (checks calibration, mount and projection)

1. Take **one photo** of the empty grid. Don't move the camera afterwards.
2. Open the photo in a viewer that shows the cursor's pixel position. For
   each cross, read the pixel at its center and type it into the `u` (across)
   and `v` (down) columns of
   [`reference_points_pixel.csv`](reference_points_pixel.csv). The `x_m` and
   `y_m` columns are already filled in for this layout; change them if you
   taped a point somewhere else.
3. Run:

   ```bash
   python testing-deployment-team/benchmarks/projection_accuracy.py \
       testing-deployment-team/benchmarks/floor_grid/reference_points_pixel.csv \
       --params camera_params.yaml
   ```

## Part B: detection mode (checks detect → project, the Week 9 prototype)

1. Put the ball on a cross so it touches the floor at the cross's center,
   step out of the picture and take a photo. Seven points:
   `C050`, `C100`, `C150`, `C200`, `L200`, `R200`, `C250`.
2. Save the photos in `floor_grid/photos/` with the names already in
   [`reference_points_ball.csv`](reference_points_ball.csv):
   `ball_C050.jpg`, `ball_C100.jpg`, and so on. `photos/` is gitignored;
   put the photos on the team drive.
3. Run:

   ```bash
   python testing-deployment-team/benchmarks/projection_accuracy.py \
       testing-deployment-team/benchmarks/floor_grid/reference_points_ball.csv \
       --params camera_params.yaml --weights detection-team/models/multiclass_v2.pt
   ```

   The script assumes the 5 cm-radius ball. If you use the 7 cm one, the
   distances will come out a few centimeters long.

## Reading the result

The script prints each point's error, the error by distance and a
PASS/FAIL line, and exits nonzero unless the check passes.

| What you see | Likely cause |
|---|---|
| Every point too far, or every point too near | Mount tilt or height is off. Re-measure both before anything else |
| Left and right swapped | A `y_m` sign is wrong (left is positive), or the image is mirrored |
| Good in the middle, worse toward `LL200` / `RR200` | Lens distortion: the calibration needs more photos near the image edges |
| Part A passes, part B doesn't | The detector's box, not the geometry: look at the photos with `detection-team/inference/visualize.py` |

For a quick look before measuring anything,
`integration-team/geometry/projection_demo.py` draws the projected grid
over the live image; its lines should land on the tape.

Record the result (errors, mount, date, PASS/FAIL) in this folder's parent
[README](../../README.md#projection-accuracy-check) and in
`docs/milestones.md`, and commit the filled-in pixel CSV with it.
