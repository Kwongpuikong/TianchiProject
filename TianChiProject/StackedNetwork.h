#ifndef STACKEDNETWORK_H
#define STACKEDNETWORK_H


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include "SparseAutoencoder.h"
#include "SparseAutoencoderActivation.h"
#include "SoftmaxRegression.h"
#include "Basefun.h"

class StackedNetwork{

public:
	StackedNetwork(std::vector<SA> _sc, SMR _smr);

	void Cost(Mat &x, Mat &y, double lambda);
	void gradientChecking(Mat &x, Mat &y, double lambda);
	void train(Mat &x, Mat &y, int batch, double lambda, double lrate, int maxIter);
	cv::Mat resultProdict(Mat &x);

private:
	std::vector<cv::Mat> scW;
	std::vector<cv::Mat> scb;
	std::vector<cv::Mat> scWg;
	std::vector<cv::Mat> scbg;
	cv::Mat smrW;
	cv::Mat smrWg;
	int nLayers;
	int nclasses;
	double cost;
};


#endif