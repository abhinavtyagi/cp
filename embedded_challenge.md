# High-Speed Stereo Block Matcher

In this challenge problem, you will improve the speed of a stereo block matcher in OpenCV.

## Stereo Block Matching Mini-Primer
Stereo vision cameras estimate the distance to an object by measuring the disparity (i.e., pixel shift, or parallax) between two image features.  The larger the disparity between the two features, the closer the object (mathematically, disparity[pixels] = camera focal length[pixels] * baseline width between cameras[m]/range[m]).  For example, hold your arm in front of you and put your thumb up (your thumb is the feature in this example).  Alternate between closing your left eye and right eye, and notice that the position of the thumb moves relative to the background (the background should be much farther away than your thumb).  As you bring your thumb closer to your nose, you will see that this shift (aka, disparity) increases.  This is the principle for estimating depth using stereo vision.

A computer estimates disparity for by iteratively comparing left to right sub-windows of an image.  Typically, the subwindow, or *block*, that is used for matching features from left to right images is 5x5, 9x9, 11x11, 15x15, or 21x21 pixels, depending on the scale and format of the image.  After the left and right images are *rectified*, then the block matching only needs to occur on the same row -- a 1D search rather than a 2D search.  Rectification means that the images are row aligned so that, for example, the corner of the same object is in the same row in the left and right rectified images.  The fancy way of saying this is that the epipolar lines are aligned.

If you want to read more about stereo block matching in plain english, (here)[https://python.plainenglish.io/the-depth-ii-block-matching-d599e9372712] is a webpage that explains the process with pretty pictures.

## Install Software
To do this challenge problem, you will need to install OpenCV.  Here are instructions for doing so:
```bash
# First clone both opencv and opencv_contrib repositories
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Now build opencv with contrib (which has cuda stereo matching)
cd opencv
mkdir build
cd build
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D OPENCV_GENERATE_PKGCONFIG=yes \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D BUILD_TIFF=ON \
    ..
make -j8  # It might take ~1 hour to build opencv
```

## Stereo Matching Example
Here is an example to build the challenge problem.
```bash
# For cmake, you may need to define the following system variable
export OpenCV_DIR=/path/to/opencv/build

# Build challenge problem for block matching
cd challenge_bm
mkdir build
cd build
cmake ..
make
./challenge_bm ../img_left.png ../img_right.png ../intrinsics.yml
```
The sample code in `main.cpp` is a simple program to show how to rectify and then compute the disparity map of a stereo image pair using `cuda::StereoBM`.  The sample code creates a `cuda::StereoBM` object called `bm`, which created to search from 0 to 255 pixels of disparity (minDisparity = 0 and numDisparities=256) for every pixel in the image.
```C++
bm = cuda::createStereoBM(256, 9);   //numDisparities=256, windowSize=9
bm->compute(left, right, disparity);
```
The main files that you will need to analyze and modify are located in the `opencv_contrib` repository:
```
opencv_contrib/modules/cudastereo/src/cuda/stereobm.hpp
opencv_contrib/modules/cudastereo/src/cuda/stereobm.cu
opencv_contrib/modules/cudastereo/src/stereobm.cpp
```

## Challenge Problem
Execution time of stereo block matching is proportional to the number of disparities searched, `numDisparities`.  For example, searching 16 pixels of disparity is 16 times faster than searching 256 pixels of disparity, which is a significant speed improvement.  Luckily, we have prior knowledge of our images that allows us to limit the search range for different regions of the image.  We know that there are no points under the road, so we do not need to search low disparity values (far away) for the road image region.  Furthermore, since there is not much motion between frames, we can use the disparity map from the previous frame to inform our search range for the current frame.  Therefore, it is valuable to limit the disparity search range for different areas of the image to speed up the stereo block matching process.  To achieve this functionality, you will make `numDisparities` input to `cv::cuda::createStereoBM()` a matrix input parameter (`cv::Mat` or `cv::cuda::GpuMat`) rather than a scalar integer input.  The complete solution should also change `minDisparity` from a scalar to matrix parameter, but for the sake of simplicity, we will ignore that parameter for this challenge problem.

Please complete the following tasks:
1. First benchmark default cuda::stereoBM performance in `main.cpp`.  Where is most of the time spent?  Is the computation limited by the memory bandwidth or by the number of cuda cores?
2. Can you speed up the opencv cuda block matcher with better choices for threads per block, register counts per thread, shared memory per block, etc.  How does this change with different GPUs, such as the Jetson Xavier NX vs. A100 vs. Quadro RTX 5000?  Can tensor cores be used to speed up performance?  Please benchmark the modified code.
3. Please modify the block matcher to take a search range for each pixel, that is, convert `cuda::createStereoBM(int numDisparities, int windowSize)` to `cuda::createStereoBM(cuda::GpuMat numDisparities, int windowSize)`.  Compare the speed of the original code with the same disparity search range for every pixel (`StereoBM(int, int)` with `numDisparities = 128`) to your new block matcher than can have different search ranges for each pixel (`StereoBM(GpuMat, int)` with `theRNG().fill(numDisparities, RNG::UNIFORM, 0, 255);`).  Both cases have the same average number of disparities searched over the entire image.
4. Benchmark your new code for `StereoBM(GpuMat, int)`.  Can you make it faster?


