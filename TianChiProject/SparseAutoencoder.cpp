#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include "SparseAutoencoder.h"
#include "Basefun.h"

#define ATD at<double>
#define IS_TEST_SA 0

using namespace std;
using namespace cv;

typedef SparseAutoencoderActivation SAA;

SparseAutoencoder::SparseAutoencoder(int _inputSize, int _hiddenSize){

	inputSize = _inputSize;
	hiddenSize = _hiddenSize;
	cost = 0.0;
	weightRandomInit(0.12);
	// for debugging...
	// cout << W1 << endl;

}

void SparseAutoencoder::weightRandomInit(double epsilon){

	double *pData;
	W1 = Mat::ones(hiddenSize, inputSize, CV_64FC1);
	for(int i=0; i < hiddenSize; i++){
	
		pData = W1.ptr<double>(i);
		for(int j=0; j < inputSize; j++){
		
			pData[j] = randu<double>();
		}
	}
	W1 = W1 * (2*epsilon) - epsilon;

	W2 = Mat::ones(inputSize, hiddenSize, CV_64FC1);
	for(int i=0; i < inputSize; i++){
	
		pData = W2.ptr<double>(i);
		for(int j=0; j < hiddenSize; j++){
		
			pData[j] = randu<double>();
		}
	}
	W2 = W2 * (2*epsilon) - epsilon;

	b1 = Mat::ones(hiddenSize, 1, CV_64FC1);
	for(int j=0; j < hiddenSize; j++){
	
		b1.ATD(j,0) = randu<double>();
	}
	b1 = b1 * (2*epsilon) - epsilon;

	b2 = Mat::ones(inputSize, 1, CV_64FC1);
	for(int j=0; j < inputSize; j++){
	
		b2.ATD(j,0) = randu<double>();
	}
	b2 = b2 * (2*epsilon) - epsilon;

	W1grad = Mat::zeros(hiddenSize, inputSize, CV_64FC1);
	W2grad = Mat::zeros(inputSize, hiddenSize, CV_64FC1);
	b1grad = Mat::zeros(hiddenSize, 1, CV_64FC1);
	b2grad = Mat::zeros(inputSize, 1, CV_64FC1);
}

SAA SparseAutoencoder::getSparseAutoencoderActivation(cv::Mat &data){

	SAA acti;
	data.copyTo(acti.aInput);
	// for debugging...
	/*
	cout << "acti.input.rows: " << acti.aInput.rows << endl
		<< "acti.input.cols: " << acti.aInput.cols << endl
		<< "W1.rows: " << W1.rows << endl
		<< "W1.cols: " << W1.cols << endl
		<< "W2.rows: " << W2.rows << endl
		<< "W2.cols: " << W2.cols << endl;
		*/

	acti.aHidden = W1 * acti.aInput + repeat(b1, 1, data.cols);
	acti.aHidden = sigmoid(acti.aHidden);
	acti.aOutput = W2 * acti.aHidden + repeat(b2, 1, data.cols);
	acti.aOutput = sigmoid(acti.aOutput);
	return acti;
}

void SparseAutoencoder::Cost(cv::Mat &data, double lambda, double sparsityParam, double beta){

	int nfeatures = data.rows;
	int nsamples = data.cols;
	// for debugging...
	// cout << "nfeatures: " << nfeatures << endl
	//	 << "nsamples: " << nsamples << endl;

	SAA acti = getSparseAutoencoderActivation(data);

	Mat errtp = acti.aOutput - data;
	pow(errtp, 2.0, errtp);
	errtp /= 2.0;
	double err = sum(errtp)[0] / nsamples;

	// pj is the mean activation of the hidden layer
	Mat pj; 
	reduce(acti.aHidden, pj, 1, CV_REDUCE_SUM);
	pj /= nsamples; 

	// err2 is the weight decay part
	Mat w1_square, w2_square;
	pow(W1, 2.0, w1_square);
	pow(W2, 2.0, w2_square);
	double err2 = sum(w1_square)[0] +sum(w2_square)[0];
	err2 *= (lambda/2.0);

	Mat err3;
	Mat temp;
	temp = sparsityParam / pj;
	log(temp, temp);
	temp *= sparsityParam;
	temp.copyTo(err3);
	temp = (1 - sparsityParam) / (1 - pj);
	log(temp, temp);
	temp *= (1 - sparsityParam);
	err3 += temp;
	cost = err + err2 + sum(err3)[0] * beta;

	Mat delta3 = -(data - acti.aOutput);
	delta3 = delta3.mul(dsigmoid(acti.aOutput));
	Mat temp2 = -sparsityParam / pj + (1-sparsityParam) / (1-pj);
	temp2 *= beta;
	Mat delta2 = W2.t() *delta3 + repeat(temp2, 1, nsamples);
	delta2 = delta2.mul(dsigmoid(acti.aHidden));
	Mat nablaW1 = delta2 * acti.aInput.t();
	Mat nablaW2 = delta3 * acti.aHidden.t();
	Mat nablab1, nablab2;
	delta3.copyTo(nablab2);
	delta2.copyTo(nablab1);
	W1grad = nablaW1 / nsamples + lambda * W1;
	W2grad = nablaW2 / nsamples + lambda * W2;
	reduce(nablab1, b1grad, 1, CV_REDUCE_SUM);
	reduce(nablab2, b2grad, 1, CV_REDUCE_SUM);
	b1grad /= nsamples;
	b2grad /= nsamples;
}

void SparseAutoencoder::updateWeights(cv::Mat w1g, cv::Mat w2g, cv::Mat b1g, cv::Mat b2g, double lrate){

	W1 -= lrate * w1g;
	W2 -= lrate * w2g;
	b1 -= lrate * b1g;
	b2 -= lrate * b2g;
}

void SparseAutoencoder::gradientChecking(cv::Mat &data, double lambda, double sparsityParam, double beta){

	Cost(data, lambda, sparsityParam, beta);

	Mat w1g (W1grad);
	// for debugging...
	// cout << w1g.ATD(0, 0) << endl;

	cout<<"test sparse autoencoder !!!"<<endl;
	double epsilon = 1e-4;
	for(int i=0; i<w1g.rows; i++){
	
		for(int j=0; j<w1g.cols; j++){
		
			double memo = W1.ATD(i,j);
			W1.ATD(i,j) = memo + epsilon;
			Cost(data, lambda, sparsityParam, beta);
			double value1 = cost;
			// for debugging...
			// cout << W1.ATD(i,j) << "\t" << cost << "\t" << value1 << endl;

			W1.ATD(i,j) = memo - epsilon;
			Cost(data, lambda, sparsityParam, beta);
			double value2 = cost;
			// for debugging...
			// cout << W1.ATD(i,j) << "\t" << cost << "\t" << value2 << endl;

			double tp = (value1 - value2) / (2*epsilon);
			cout << i << ", " << j << ", " << tp << ", " << w1g.ATD(i,j)
				<< ", " << w1g.ATD(i,j) / tp << endl;
			W1.ATD(i,j) = memo;
		}
	}
}

void SparseAutoencoder::train(cv::Mat &data, int batch, double lambda, double sparsityParam, double beta, double lrate, int maxIter){

	int nfeatures = data.rows;
	int nsamples = data.cols;
	weightRandomInit(0.12);
	// for debugging...
	/*
	cout << "Training sc..."
		<< "input_size of sc: " << inputSize << endl
		<< "hidden_size of sc: " << hiddenSize << endl;
		*/

	if(IS_TEST_SA){
	
		gradientChecking(data, lambda, sparsityParam, beta);
	}
	else{
	
		int converge = 0;
		double lastcost = 0.0;
		cout << "\t starting training....\n" << endl;
		while(converge < maxIter){
		
			int randomNum = rand() % (data.cols - batch);

			Rect roi = cv::Rect(randomNum, 0, batch, data.rows);
			Mat batchX = data(roi);
			Cost(batchX, lambda, sparsityParam, beta);
			cout << "learning step: " << converge
				<< ", cost function value = " << cost
				<< ", randomNum = " << randomNum << endl;
			if((fabs(cost - lastcost) <= 5e-5) && (converge > 0)) break;
			if(cost <= 0.0) break;
			lastcost = cost;
			W1 -= lrate * W1grad;
			W2 -= lrate * W2grad;
			b1 -= lrate * b1grad;
			b2 -= lrate * b2grad;
			++converge;
		}
	}
}


Mat SparseAutoencoder::getW1(){

	 // Mat res;
	 // W1.copyTo(res);
	 // return res;
	return W1;
}

Mat SparseAutoencoder::getW2(){

	// Mat res;
	// W2.copyTo(res);
	// return res;
	return W2;
}

Mat SparseAutoencoder::getb1(){

	// Mat res;
	// b1.copyTo(res);
	// return res;
	return b1;
}

Mat SparseAutoencoder::getb2(){

	// Mat res;
	// b2.copyTo(res);
	// return res;
	return b2;
}

int SparseAutoencoder::getinputSize(){

	return inputSize;
}

int SparseAutoencoder::gethiddenSize(){

	return hiddenSize;
}



