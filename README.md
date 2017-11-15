# GaussianBlur using CUDA and OpenCV

Applying gaussian blur method and using parallelized kernels (in CUDA + OpenCV environment), reducing the noise in an image.

<img src="/balloons.png" alt="Original" height="318" />&nbsp;&nbsp;->&nbsp;&nbsp;<img src="/output.png" alt="Blurred" height="318"/> <br />

### Objectives<br />
- Convert RGBA to Array of Structures (AoS) <br />
- Apply Gaussian Blur method to each channel <br />
- Parallelize using kernels <br />
