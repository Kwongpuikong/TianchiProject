#ifndef STACKEDNETWORK_H
#define STACKEDNETWORK_H


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include "SparseAutoencoder.h"
#include "SoftmaxRegression.h"


class StackedNetwork{

public:
	StackedNetwork(std::vector<SparseAutoencoder> _sc, SoftmaxRegression _smr);

	void Cost(cv::Mat &x, cv::Mat &y, double lambda);
	void gradientChecking(cv::Mat &x, cv::Mat &y, double lambda);
	void train(cv::Mat &x, cv::Mat &y, int batch, double lambda, double lrate, int maxIter);
	cv::Mat resultProdict(cv::Mat &x);

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