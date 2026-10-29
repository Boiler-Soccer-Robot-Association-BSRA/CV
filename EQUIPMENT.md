# Equipment Checklist

Physical items the CV team needs this semester, roughly in the order
they become blocking. Check items off as they're acquired. Add a note
with who's responsible or the purchase status.

## Needed before November (camera calibration milestone)

| Item | Notes | Owner / Status |
|---|---|---|
| [ ] **Camera matching the robot's onboard camera** | **Chosen: [e-con See3CAM_24CUG](https://www.e-consystems.com/industrial-cameras/ar0234-usb3-global-shutter-camera.asp)** (onsemi AR0234, 1920×1200, global shutter, 60 fps at full res / 120 fps at HD, USB 3.1 Type-C, UVC). M12 (S-mount) lens; e-con's lens is very wide: 104.6° horizontal, 61.6° vertical, 128.2° diagonal. Board 30×30 mm (35.3×35.3 mm in the enclosure). Things to get right: (1) order the **exact lens the robot will use**: calibration is only valid for that camera + lens; (2) after focusing, **lock the M12 lens** (locknut or a dab of glue), since turning it invalidates the calibration; (3) **calibrate at the resolution we'll run at**; (4) at ~128° diagonal, the standard OpenCV model may not fit; try `cv2.fisheye` or `CALIB_RATIONAL_MODEL` if reprojection error is above ~0.5 px; (5) check whether the enclosed version has a 1/4″-20 tripod thread (not listed on the product page); the bare board needs a clamp or 3D-printed holder. | Being ordered |
| [ ] **Printed checkerboard calibration pattern** | 10×7 squares = **9×6 inner corners**, **35 mm** squares (board 350×245 mm), printed on **11×17″ (Tabloid) or A3**. A big board matters with this wide lens; A4/Letter is too small except for testing code. Print at **100% / Actual size** (no "fit to page"), matte paper, then check the printed 100 mm scale line and one square with a ruler. Mount it **perfectly flat** on foam board (or have a print shop print it on foam board). Write the measured square size on the back. PDFs (Tabloid, A3, Letter test version) are ready, generated to exact size and checked with `cv2.findChessboardCorners`; ask Hossein. | PDF ready, not printed |
| [ ] **Simple tripod or fixed mount** | For consistent, repeatable shots of the checkerboard at different angles during calibration. Any small tripod with a standard **1/4″-20** screw and a tilting/ball head. Best if it can be set to the robot's camera height, so Week 8's ground-plane projection can be tested on the floor before the robot exists. Camera attachment: see the camera row. | |

## Worth considering, not yet blocking

| Item | Notes | Owner / Status |
|---|---|---|
| [ ] **GPU/compute access** | Not physical equipment, but worth budgeting for if training is bottlenecked by laptop hardware. Check cloud credits (e.g. Colab Pro) or university compute access for student orgs before buying hardware. | |

## Later (December robustness testing), don't buy yet

| Item | Notes | Owner / Status |
|---|---|---|
| [ ] **Real-world test props** (varied lighting setup, RoboCup-style ball, etc.) | Hold off until testing-deployment-team actually needs real-world edge cases beyond what TORSO-21 covers. Revisit this section in November once detection is working. | |

---

*Last updated: 2026-10-29. Update the status of each item as it changes — this doc is meant to stay current, not be written once and forgotten.*
