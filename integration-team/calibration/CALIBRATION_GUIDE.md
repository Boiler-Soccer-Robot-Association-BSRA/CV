# Camera calibration guide

How to calibrate the See3CAM_24CUG using the checkerboard and tripod.

**Calibration** measures how this exact camera and lens bend the image. The
result (`camera_params`) lets `geometry/projection.py` turn a pixel into a real
position on the ground. Without it, positions will be wrong, especially near
the image edges of this very wide lens.

*Last updated: 2026-11-01. Owner: integration-team.*

## What you need

| Item | Details | Status |
|---|---|---|
| Camera | e-con See3CAM_24CUG, 1920×1200, M12 lens, USB-C | Being ordered |
| Lens | The exact lens the robot will use | Confirm with e-con |
| Checkerboard | 10×7 squares = **9×6 inner corners**, 35 mm squares | PDF ready, not printed |
| Tripod | Amazon 50-inch tripod, 16.1–50″, 3-way head, quick-release plate | Being bought |
| Camera holder | Only needed if the camera has no 1/4″-20 hole. Check e-con's "See3CAM Mounting Document" | Unknown |
| Laptop | USB 3 port and this repo set up with `./scripts/setup.sh` | |

## Step 1: Print the checkerboard

Use `checkerboard_Tabloid_35mm.pdf`. Ask Hossein for the file.

1. Add money to your **Paid Supplement** quota in PaperCut. The default print
   quota doesn't work on the wide-format printers.
2. Go to the Purdue IT Service Desk at **WALC (1st floor)** or **HSSE** during
   staffed hours, at least 30 minutes before closing.
3. Tell the staff: **"Print at 100%, actual size, no scaling, don't fit it to
   the roll."** The page size is 17″ × 11″, landscape.
4. Ask for matte paper if they have it. Glare hides the corners.
5. Before leaving, measure the printed **100 mm line** and one **35 mm square**.
   If either is wrong, reprint it.
6. Mount it **perfectly flat** on foam board or stiff cardboard, with no bubbles
   or curling.
7. Write on the back: `9x6 inner corners, ___ mm squares`, using your measurement.

## Step 2: Set up the camera and tripod

1. Attach the lens and connect the camera over USB.
2. Focus at the distance the robot needs to see, then **lock the lens** with a
   locknut or a dab of glue.
3. Mount the camera on the tripod. Tape the USB cable to a tripod leg so it
   can't pull the tripod.
4. Choose the resolution you'll use on the robot. Calibration only applies to
   that resolution, so write it down.
5. Use bright, even light. Avoid direct sun and reflections on the board.

**If the lens is turned, removed or swapped, calibrate again.**

## Step 3: Take the photos

Keep the tripod still and move the board. Take **20–30 photos**, and hold the
board completely still for each one.

Include:

- [ ] Board close to the camera
- [ ] Board far from the camera
- [ ] Board tilted left, right, up and down
- [ ] Board near **all four corners** of the picture
- [ ] Board along all four edges
- [ ] Board in the center

For every photo, check that:

- All corners of the board are in view.
- The photo is sharp, with no blur or glare.
- Your fingers don't cover any squares.

Corners and edges matter most because that's where this lens bends the image
the most. Save the photos in one folder, such as `~/calibration_photos/`. Don't
commit the photos to git.

## Step 4: Run the calibration

`calibrate.py` finds the board in each photo, skips photos where it isn't found,
and prints how many it used. It then prints the reprojection error and saves the
parameters and resolution to `--output`. It has only been tested on computer-made
board images so far, not real photos.

Pass the **measured** square size in mm with `--square-mm` (35.0 if the print is
exact).

Run it like this:

```bash
python integration-team/calibration/calibrate.py ~/calibration_photos --output camera_params.yaml --square-mm 35.0
```

## Step 5: Check the result

1. **Reprojection error:** this shows how well the result fits the photos, in
   pixels. Aim for **under about 0.5 px**.
2. **Straight lines:** undistort a photo with `cv2.undistort`. Straight real-world
   lines, such as door frames, should look straight, including near the edges.
3. **Real distance:** after Week 8's projection code, put an object at a measured
   spot on the floor and compare it with the computed position.

If the error is too high:

1. Remove blurry photos and photos where the board was found poorly.
2. Add more photos near the corners and edges.
3. Check that the board is flat and the square size is correct.
4. Try `cv2.CALIB_RATIONAL_MODEL`, or `cv2.fisheye.calibrate`. At about 128°
   diagonal, the standard model may not fit this lens well.

## Step 6: Next steps

1. Record the camera, lens, resolution, square size, date and error with the
   saved parameters.
2. Hand the parameters to `geometry/projection.py`. It also needs where the
   camera sits on the robot, which calibration can't measure. Measure it by hand
   and add a `mount:` section to the saved YAML (meters and degrees; pitch is
   positive when the camera tilts down):

   ```yaml
   mount:
     height_m: 0.45   # lens height above the floor
     pitch_deg: 20.0
     roll_deg: 0.0
     yaw_deg: 0.0
     x_m: 0.0         # lens position ahead of the robot's origin
     y_m: 0.0         # lens position left of the robot's origin
   ```

   Then `load_camera_params("camera_params.yaml")` reads both. Pitch and height
   errors matter most: about 1° and 2 cm moves a point at 2 m by roughly 6 cm.
3. For Week 8, set the tripod to the robot's camera height and tilt, and test on
   the floor. Height and tilt are measured separately from this calibration.
4. Moving the camera to the robot doesn't require a new calibration, as long as
   the lens and focus haven't changed.

## Quick reminders

- Use the same camera, lens, focus and resolution as the robot.
- Print at 100% and measure the printed squares.
- Keep the board flat and still.
- Get the corners and edges of the picture.
- Recalibrate if anything about the lens changes.
