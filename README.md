# Harris Corner Detection

Implementation of classic computer vision algorithms from scratch using NumPy.
Course: Computer Vision

## Part A: Image Convolutions

Implemented `convolutionMask(img, mask)` — 2D convolution with side-by-side display of original and convolved image. Defined four convolution kernels: a 5×5 averaging kernel, a pattern-matching kernel for a specific shape, a pattern-matching kernel with don't-care values, and a 3-pixel rightward shift kernel.

## Part B: Harris Corner Detector

Implemented the full Harris corner detection pipeline:

- `deriv_gauss_xy(s_smooth)` — derivative-of-Gaussian kernels for x and y directions
- `grad_xy(im, s_smooth)` — image gradients Ix and Iy via convolution
- `harris_corner(im, s_smooth, s_neighb, th, density_size, display)` — structure tensor H per pixel, corner strength via `det(H) - k·trace(H)²`, thresholding, and non-maximum suppression

Tested on a synthetic image, a checkerboard, and a real-world scene. Explored the effect of all four parameters on detection quality.

## Tools

Python 3.12, NumPy 2.0.2, SciPy, OpenCV, Matplotlib
