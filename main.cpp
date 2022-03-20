#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudastereo.hpp>

using namespace cv;
using namespace std;

// Example usage:
// ./challenge ../img_left.png ../img_right.png ../intrinsics.yml
int main(int argc, char* argv[])
{
    cout << "OpenCV Version: " << CV_VERSION << endl;

	// Read left image file
    Mat left;
    left = imread( argv[1], 1 );

	// Read right image file
    Mat right;
    right = imread( argv[2], 1 );

	// Read camera parameters file
	int height, width;
	cv::FileStorage opencv_file( argv[3], cv::FileStorage::READ );
	cv::Mat K1, K2, distCoeffs1, distCoeffs2, R, T;
	opencv_file["K1"] >> K1;
	opencv_file["K2"] >> K2;
	opencv_file["distCoeffs1"] >> distCoeffs1;
	opencv_file["distCoeffs2"] >> distCoeffs2;
	opencv_file["R"] >> R;
	opencv_file["T"] >> T;
	opencv_file["height"] >> height;
	opencv_file["width"] >> width;
	opencv_file.release();

    // Create rectification maps
    Mat R1, R2, P1, P2, Q; // outputs 
	stereoRectify(K1, distCoeffs1, K2, distCoeffs2,
            Size(width, height), R, T, R1, R2, P1, P2, Q);

	Mat map1x, map1y; // outputs
    initUndistortRectifyMap(K1, distCoeffs1, R1, P1, 
            Size(width, height), CV_32FC1, map1x, map1y);

    Mat map2x, map2y; // outputs
    initUndistortRectifyMap(K2, distCoeffs2, R2, P2, 
            Size(width, height), CV_32FC1, map2x, map2y);

	// Rectify input images
	Mat left_rectified, right_rectified;  // outputs

    cv::remap(left, left_rectified, map1x, map1y, cv::INTER_LINEAR, 
		cv::BORDER_CONSTANT, 0);

    cv::remap(right, right_rectified, map2x, map2y, cv::INTER_LINEAR, 
		cv::BORDER_CONSTANT, 0);

	// Convert rectified images from RGB to grayscale
	cv::Mat left_rectified_gray, right_rectified_gray;
	cv::cvtColor(left_rectified, left_rectified_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(right_rectified, right_rectified_gray, cv::COLOR_BGR2GRAY);

	//
	// Compute disparity map
	//

	// Create StereoBM object
	int numDisparities = 256;
	int windowSize = 9;
	cv::Ptr<cuda::StereoBM> bm;
	bm = cuda::createStereoBM(numDisparities, windowSize);

	// Run stereo matching algorithm
	cuda::GpuMat left_rectified_gray_g( left_rectified_gray.size(), CV_8U );
	cuda::GpuMat right_rectified_gray_g( right_rectified_gray.size(), CV_8U );
	cuda::GpuMat disparity_g( left_rectified_gray.size(), CV_8U );
	left_rectified_gray_g.upload( left_rectified_gray );
	right_rectified_gray_g.upload( right_rectified_gray );
	bm->compute(left_rectified_gray_g, right_rectified_gray_g, disparity_g);

	// Save output
	Mat disparity;
	disparity_g.download(disparity);
	imwrite("disparity.png", disparity);

    return 0;
}
