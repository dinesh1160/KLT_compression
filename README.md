# Karhunen-Loève Transform (KLT) based image compression

Karhunen-Loève Transform (KLT) using Python and OpenCV — It allows you to load any image, apply KLT to compress it by preserving 95% of the image's energy, reconstruct the image, and save the result.

## Features

- Accepts any .png, .jpg, .jpeg, or .bmp images
- converts images to grayscale
- Retains 95% variance
- Displays:
  - Original image
  - Reconstructed image
  - Difference image
- Prints PSNR (quality metric)


## Requirements

Python 3.6+ and the following Python packages:

```bash
pip install numpy opencv-python matplotlib
```
install tkinker

```bash
sudo apt install python3-tk
```
