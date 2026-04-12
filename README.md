# Harris Corner Detection
Implementation of classic computer vision algorithms from scratch using NumPy.

## Assignment: HW1 — Image Convolutions & Harris Corner Detector

### Part A: Image Convolutions (complete)
- Implemented `convolutionMask(img, mask)` — performs 2D convolution and displays original vs. convolved image side by side.
- Defined four convolution kernels:
  - `mask1`: 5×5 averaging kernel (blur)
  - `mask2`: pattern-matching kernel for a specific shape
  - `mask3`: pattern-matching kernel with don't-care values
  - `mask4`: 3-pixel rightward shift kernel

### Part B: Harris Corner Detector (complete, pending final review)
- Implemented `deriv_gauss_xy(s_smooth)` — builds 2D derivative-of-Gaussian kernels for x and y directions
- Implemented `grad_xy(im, s_smooth)` — computes image gradients Ix and Iy via convolution
- Implemented `harris_corner(im, s_smooth, s_neighb, th, density_size, display)` including:
  - Structure tensor H per pixel via Gaussian-weighted gradient products
  - Corner strength via `det(H) - k·trace(H)²`
  - Thresholding and non-maximum suppression via `maximum_filter`
  - Display of original, gradients, corner response, and detected corners
- Tested on synthetic image (white rectangle), checkerboard, and `view6_crop.tif`
- Explored effects of all four parameters: `s_smooth`, `s_neighb`, `th`, `density_size`
- Pending: final parameter tuning on `view6_crop.tif` and scatter plot axis fix

## Tools
Python 3.12, NumPy 2.0.2, SciPy, OpenCV, Matplotlib
