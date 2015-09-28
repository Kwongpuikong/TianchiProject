#ifndef SOFTMAXREGRESSION_H
#define SOFTMAXREGRESSION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>


/*softmax regression layer*/
class SoftmaxRegression{

public:
	SoftmaxRegression(){};
	SoftmaxRegression(int _inputSize, int _nclasses);
	
	void copyfrom(SoftmaxRegression smr);
	void weightRandomInit(double epsilon);
	void Cost(cv::Mat &x, cv::Mat &y, double lambda);
	void gradientChecking(cv::Mat &x, cv::Mat &y, double lambda);
	void train(cv::Mat &x, cv::Mat &y, int batch, double lambda, double lrate, int maxIter);
	int getnclasses();
	int getinputSize();
	double getcost();
	cv::Mat getWeight();
	cv::Mat getWgrad();

private:
	cv::Mat Weight;
	cv::Mat Wgrad;
	int inputSize;
	int nclasses;
	double cost;
};

#endif