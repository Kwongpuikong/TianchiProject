#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include "SoftmaxRegression.h"


#define ATD at<double>
#define IS_TEST_SMR 0

using namespace std;
using namespace cv;

SoftmaxRegression::SoftmaxRegression(int _inputSize, int _nclasses){

	inputSize = _inputSize;
	nclasses = _nclasses;
	weightRandomInit(0.12);
}

void SoftmaxRegression::copyfrom(SMR smr){

	Weight = getWeight();
	Wgrad = getWgrad();
	inputSize = getinputSize();
	nclasses = getnclasses();
	cost = getcost();
}

void SoftmaxRegression::weightRandomInit(double epsilon){

	Weight = Mat::ones(nclasses, inputSize, CV_64FC1);
	double *pData;
	for(int i=0; i < Weight.rows; i++){
	
		for(int j=0; j < Weight.cols; j++){
		
			pData[j] = randu<double>();
		}
	}
	Weight = Weight * (2*epsilon) - epsilon;
	cost = 0.0;
	Wgrad = Mat::zeros(nclasses, inputSize, CV_64FC1);
}

void SoftmaxRegression::Cost(Mat &x, Mat &y, double lambda){

	int nsampeles = x.cols;
	int nfeatures = x.rows;
	Mat theta(Weight);
	Mat M = theta * x;
	// temp, temp2 forbid the overflow
	Mat temp, temp2;
	temp = Mat::ones(1, M.cols, CV_64FC1);
	reduce(M, temp, 0, CV_REDUCE_SUM);
	temp2 = repeat(temp, nclasses, 1);
	M -= temp2;
	exp(M,M);
	// temp, temp2 here is for calcs
	temp = Mat::ones(1, M.cols, CV_64FC1);
	reduce(M, temp, 0, CV_REDUCE_SUM);
	temp2 = repeat(temp, nclasses, 1);
	divide(M, temp2, M);

	Mat groundTruth = Mat::zeros(nclasses, nsampeles, CV_64FC1);
	for(int i=0; i<nsampeles; i++){
	
		groundTruth.ATD(y.ATD(0,i),i) = 1.0;
	}
	// calc cost
	Mat logM;
	log(M, logM);
	temp = groundTruth.mul(logM);
	cost = -sum(temp)[0] / nsampeles;
	Mat theta2;
	pow(theta, 2.0, theta2);
	cost += sum(theta2)[0] *lambda /2;
	// calc grad
	temp = groundTruth - M;
	temp = temp * x.t();
	Wgrad = -temp / nsampeles;
	Wgrad += lambda * theta;
}

void SoftmaxRegression::gradientChecking(Mat &x, Mat &y, double lambda){

	Cost(x, y, lambda);
	Mat grad(Wgrad);
	cout << "test softmax regression!!!" << endl;
	double epsilon = 1e-4;
	for(int i=0; i < Weight.rows; i++){
	
		for(int j=0; j < Weight.cols; j++){
		
			double memo = Weight.ATD(i,j);
			Weight.ATD(i,j) = memo + epsilon;
			Cost(x,y,lambda);
			double value1 = cost;
			Weight.ATD(i,j) = memo - epsilon;
			Cost(x,y,lambda);
			double value2 = cost;
			double tp = (value1 - value2) / (2 * epsilon);
			cout << i << ", " << j << ", " << tp << ", " << grad.ATD(i, j)
				<< ", " << grad.ATD(i, j) / tp << endl;
            Weight.ATD(i, j) = memo;
		}
	}
}

void SoftmaxRegression::train(Mat &x, Mat &y, int batch, double lambda, double lrate, int maxIter){

	int nfeatures = x.rows;
	int nsamples = x.cols;
	weightRandomInit(0.12);
	if(IS_TEST_SMR){
	
		gradientChecking(x, y, lambda);
	}
	else{
	
		int converge = 0;
		double lastcost = 0.0;
		cout << "softmax regression learning..." << endl;
		while(converge < maxIter){
		
			int randomNum = rand() % (x.cols - batch);
			Rect roi = Rect(randomNum, 0, batch, x.rows);
			Mat batchX = x(roi);
			roi = Rect(randomNum, 0, batch, y.rows);
			Mat batchY = y(roi);

			Cost(batchX, batchY,lambda);
			cout<<"learning step: "<<converge<<", Cost function value = "<<cost<<", randomNum = "<<randomNum<<endl;
			if((fabs(cost-lastcost)<=1e-6) && converge > 0) break;
			if(cost <= 0) break;
			lastcost = cost;
			Weight -= lrate * Wgrad;
			++ converge;
		}
	}
}

int SoftmaxRegression::getnclasses(){return nclasses;}

int SoftmaxRegression::getinputSize(){return inputSize;}

Mat SoftmaxRegression::getWeight(){ Mat res; Weight.copyTo(res); return res; }

Mat SoftmaxRegression::getWgrad(){ Mat res; Wgrad.copyTo(res); return res; }

double SoftmaxRegression::getcost(){ return cost; }