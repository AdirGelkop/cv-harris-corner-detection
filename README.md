# Harris Corner Detection

Implementation of classic computer vision algorithms from scratch using NumPy.

## Assignment: HW1 — Image Convolutions & Harris Corner Detector

### Part A: Image Convolutions (in progress)
- Implemented `convolutionMask(img, mask)` — performs 2D convolution and displays original vs. convolved image side by side.
- Defined four convolution kernels:
  - `mask1`: 5×5 averaging kernel (blur)
  - `mask2`: pattern-matching kernel for a specific shape
  - `mask3`: pattern-matching kernel with don't-care values
  - `mask4`: 3-pixel rightward shift kernel

### Part B: Harris Corner Detector (not started)
- Implement `harris_corner()` including:
  - Image derivative kernels (`deriv_gauss_xy`)
  - Gradient computation (`grad_xy`)
  - Structure tensor matrix H per pixel
  - Corner strength via minimum eigenvalue or det/trace formula
  - Thresholding and non-maximum suppression
- Apply and explore parameters on synthetic and real images
- Analyze results on `view6_crop.tif`

## Tools
Python 3.12, NumPy 2.0.2, SciPy, OpenCV, Matplotlib
