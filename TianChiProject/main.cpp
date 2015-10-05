#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include "Basefun.h"
#include "SoftmaxRegression.h"
#include "SparseAutoencoder.h"
#include "SparseAutoencoderActivation.h"
#include "StackedNetwork.h"

using namespace std;
using namespace cv;

typedef SparseAutoencoder SA;
typedef SparseAutoencoderActivation SAA;
typedef SoftmaxRegression SMR;

int main(){

	long start;
	start = clock();

	Mat trainX, trainY;
	Mat testX, testY;
	readData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000);
	readData(testX, testY, "mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", 10000);

	// use the first 500 training samples, for testing the algorithm.
	// Rect roi = cv::Rect(0, 0, 60000, trainX.rows);
	// trainX = trainX(roi);
	// roi = cv::Rect(0, 0, 60000, trainY.rows);
	// trainY = trainY(roi);

	cout << "Train: \n"
		<< "trainX.rows: " << trainX.rows << endl
		<< "trainX.cols: " << trainX.cols << endl
		<< "trainY.rows: " << trainY.cols << endl;

	cout << "Test: \n"
		<< "testX.rows: " << testX.rows << endl
		<< "testX.cols: " << testX.cols << endl
		<< "testY.rows: " << testY.cols << endl;

	// batch is for SGD.
	int batch = trainX.cols / 100;
	// lambda is for weight decay.
	double lambda = 3e-3;
	// sparsityparam is for sparsity cost.
	double sparsityParam = 0.1;
	// beta is for sparsity cost weights.
	double beta = 3.0;
	// learning_rate is the step size for SGD.
	double learning_rate = 2e-2;
	// maxIter is the max num of iterations.
	int maxIter = 2;
	// num_of_sa is the layers of stacked auto encoder. 
	int num_of_sa = 2;
	// input_size is a vector implies the input_size of each sa.
	int tmp_input_size[] = { 28 * 28, 800 };
	vector<int> input_size(tmp_input_size, tmp_input_size + num_of_sa);
	// hidden_size is the numbers of a feature.
	int tmp_hidden_size[] = { 800, 1000 };
	vector<int> hidden_size(tmp_hidden_size, tmp_hidden_size + num_of_sa);

	// vector<SA> stackedcoding;
	vector<Mat> activations;

	vector<Mat> stackedW;
	vector<Mat> stackedb;

	for (int i = 0; i < num_of_sa; i++){
		
		cout << "Training no." << (i + 1) << " sc..." << endl;

		Mat tempX;
		if (i == 0){
		
			// trainX.copyTo(tempX);
			tempX = trainX.clone();

		}
		else{
		
			activations[i - 1].copyTo(tempX);
		}
	
		SA tmpsa(input_size[i], hidden_size[i]);
		
		tmpsa.train(tempX, batch, lambda, sparsityParam, beta, learning_rate, maxIter);

		Mat tmpacti = tmpsa.getW1() * tempX + repeat(tmpsa.getb1(), 1, tempX.cols);
		tmpacti = sigmoid(tmpacti);
		activations.push_back(tmpacti);

		// stackedcoding.push_back(tmpsa);
		stackedW.push_back(tmpsa.getW1());
		stackedb.push_back(tmpsa.getb1());
		// activations.push_back(tmpacti);
		
	}
	// release memory.
	activations.clear();

	// !! The above pass debug.

	// smr_input_size is the size of a feature.
	int smr_input_size = 600;
	// smr_nclasses is the num of classes.
	int smr_nclasses = 10;
	// smr_lambda is the weight decay.
	double smr_lambda = 3e-3;
	// smr_learning_rate is the setp size of SGD.
	double smr_learning_rate = 2e-2;
	// smr_maxIter is the max iterations.
	int smr_maxIter = 10;
	SMR smr(smr_input_size, smr_nclasses);
	smr.train(activations[activations.size() - 1], trainY, batch, smr_lambda, smr_learning_rate, smr_maxIter);

	// construct a stacked network
	// StackedNetwork sc(stackedcoding, smr);
	StackedNetwork sc(stackedW, stackedb, smr);
	

	// sc_lambda is for sc.train
	double sc_lambda = 1e-4;
	// sc_learning_rate is the step size for SGD.
	double sc_learning_rate = 2e-2;
	// sc_maxIter is the max iterations.
	int sc_maxIter = 10;
	sc.train(trainX, trainY, batch, sc_lambda, sc_learning_rate, maxIter);
	Mat result = sc.resultProdict(testX);

	Mat err(testY);
	err -= result;
	int correct = err.cols;
	for (int i = 0; i < err.cols; i++){
	
		if (err.at<double>(0, i) != 0){
		
			--correct;
		}
	}
	cout << "correct: " << correct << ", total: " << err.cols
		<< ", accuracy: " << double(correct) / (double)(err.cols) 
		<< endl;

	ofstream ofs;
	ofs.open("log.txt", ofstream::out);
	ofs << "correct: " << correct << endl
		<< "total: " << err.cols << endl
		<< "accuracy: " << double(correct) / (double)(err.cols)
		<< endl;

	return 0;
}