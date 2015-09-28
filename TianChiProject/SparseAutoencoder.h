#ifndef SPARSEAUTOENCODER_H
#define SPARSEAUTOENCODER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include "SparseAutoencoderActivation.h"
#include "SoftmaxRegression.h"
#include "Basefun.h"

/*A two hidden layers sparse auto encoder*/
class SparseAutoencoder{
	
public:
	SparseAutoencoder(int _inputSize, int _hiddenSize);
	

	// initial the weights of sparseAutoencoder
	void weightRandomInit(double epsilon);
	SAA getSparseAutoencoderActivation(cv::Mat &data);
	void Cost(cv::Mat &data, double lambda, double sparsityParam, double beta);
	void updateWeights(cv::Mat w1g, cv::Mat w2g, cv::Mat b1g, cv::Mat b2g, double lrate);
	void gradientChecking(cv::Mat &data, double lambda, double sparsityParam, double beta);
	void train(cv::Mat &data, int batch, double lambda, double sparsityParam, double beta, double lrate, double maxIter);
	cv::Mat getW1();
	cv::Mat getW2();
	cv::Mat getb1();
	cv::Mat getb2();
	int getinputSize();
	int gethiddenSize();

private:
	int inputSize;
	int hiddenSize;
	cv::Mat W1;
	cv::Mat W2;
	cv::Mat b1;
	cv::Mat b2;
	cv::Mat W1grad;
	cv::Mat W2grad;
	cv::Mat b1grad;
	cv::Mat b2grad;
	double cost;
};

typedef SparseAutoencoder SA;

#endif