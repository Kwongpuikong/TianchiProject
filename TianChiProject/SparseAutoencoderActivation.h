#ifndef SPARSEAUTOENCODERACTIVITION_H
#define SPARSEAUTOENCODERACTIVITION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>

/*sparse auto encoder activation*/
struct SparseAutoencoderActivation{

	cv::Mat aInput;
	cv::Mat aHidden;
	cv::Mat aOutput;
};


#endif