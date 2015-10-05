#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include "StackedNetwork.h"
#include "Basefun.h"

#define ATD at<double>
#define IS_TEST_FT 0

using namespace std;
using namespace cv;

typedef SparseAutoencoder SA;
typedef SoftmaxRegression SMR;

StackedNetwork::StackedNetwork(vector<SA> _sc, SMR _smr){

	nLayers = _sc.size();
	nclasses = _smr.getnclasses();

	for(int i=0; i < nLayers; i++){

		Mat tmp = _sc[i].getW1();
		int rows = tmp.rows;
		int cols = tmp.cols;
		// for debugging...
		// cout << rows << ", " << cols << endl;
		scW.push_back(tmp);
		scWg.push_back(Mat::zeros(rows,cols,CV_64FC1));

		Mat tmp2 = _sc[i].getb1();
		scb.push_back(tmp2);
		scbg.push_back(Mat::zeros(tmp2.size(),CV_64FC1));
	}

	smrW = _smr.getWeight();
	smrWg = Mat::zeros(smrW.size(),CV_64FC1);

	cost = 0.0;
}

StackedNetwork::StackedNetwork(vector<Mat> _scW, vector<Mat> _scb, SMR _smr){

	nLayers = _scW.size();
	nclasses = _smr.getnclasses();

	for (int i = 0; i < nLayers; i++){

		Mat tmp = _scW[i];
		int rows = tmp.rows;
		int cols = tmp.cols;
		scW.push_back(tmp);
		scWg.push_back(Mat::zeros(rows, cols, CV_64FC1));

		tmp = _scb[i];
		scb.push_back(tmp);
		scbg.push_back(Mat::zeros(tmp.size(), CV_64FC1));
	}

	smrW = _smr.getWeight();
	smrWg = Mat::zeros(smrW.size(), CV_64FC1);

	cost = 0.0;
}


void StackedNetwork::Cost(Mat &x, Mat &y, double lambda){

	int nfeatures = x.rows;
	int nsamples = x.cols;
	vector<Mat> acti;

	acti.push_back(x);
	for(int i=1; i <= nLayers; i++){
	
		Mat tmpacti = scW[i-1] * acti[i-1] + repeat(scb[i-1], 1, x.cols);
		acti.push_back(sigmoid(tmpacti));
	}
	Mat M = smrW * acti[acti.size() - 1];
	Mat tmp;
	reduce(M, tmp, 0, CV_REDUCE_MAX);
	M = M - repeat(tmp, M.rows, 1); // ?
	Mat p;
	exp(M, p);
	reduce(p, tmp, 0, CV_REDUCE_SUM);
	divide(p, repeat(tmp, p.rows, 1), p);
	Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
	for(int i=0; i<nsamples; i++){
	
		groundTruth.ATD(y.ATD(0,i),i) = 1.0;
	}
	Mat logP;
	log(p,logP);
	logP = logP.mul(groundTruth);
	cost = -sum(logP)[0] / nsamples;
	pow(smrW, 2.0, tmp);
	cost += sum(tmp)[0] * lambda/2;

	tmp = (groundTruth - p) * acti[acti.size()-1].t();
	tmp /= -nsamples; 
	smrWg = tmp + lambda * smrW;

	vector<Mat> delta(acti.size());
	delta[delta.size()-1] = - smrW.t() * (groundTruth-p);
	delta[delta.size()-1] = delta[delta.size()-1].mul(dsigmoid(acti[acti.size()-1]));
	for(int i = delta.size()-2; i>=0; i--){
	
		delta[i] = scW[i].t() * delta[i+1];
		delta[i] = delta[i].mul(dsigmoid(acti[i]));
	}
	for(int i=nLayers-1; i>=0; i--){
	
		scWg[i] = delta[i+1] * acti[i].t();
		scWg[i] /= nsamples;
		reduce(delta[i+1], tmp, 1, CV_REDUCE_SUM);
		scbg[i] = tmp / nsamples;
	}
	acti.clear();
	delta.clear();
}

void StackedNetwork::gradientChecking(Mat &x, Mat &y, double lambda){

	Cost(x, y,lambda);
	Mat grad(scWg[1]);
	cout<<"test fine-tune network !!!!"<<endl;
	double epsilon = 1e-4;
	for(int i=0; i<scWg[1].rows; i++){
	
		for(int j=0; j<scWg[1].cols; j++){
		
			double memo = scWg[1].ATD(i,j);
			scWg[1].ATD(i, j) = memo + epsilon;
			Cost(x, y, lambda);
			double value1 = cost;
			scWg[1].ATD(i, j) = memo - epsilon;
			Cost(x, y, lambda);
			double value2 = cost;
			double tp = (value1 - value2) / (2*epsilon);
			cout<< i << ", " << j << ", " << tp << ", " << grad.ATD(i, j)
				<< ", " << grad.ATD(i, j) / tp << endl;
			scWg[1].ATD(i, j) = memo;
		}
	}
}

void StackedNetwork::train(Mat &x, Mat &y, int batch, double lambda, double lrate, int maxIter){

	if (IS_TEST_FT){
        gradientChecking(x, y, lambda);
    }
	else{
        
		int converge = 0;
        double lastcost = 0.0;
        cout<<"Fine-Tune network Learning: "<<endl;
        while(converge < maxIter){
 
            int randomNum = rand() % (x.cols - batch);
            Rect roi = Rect(randomNum, 0, batch, x.rows);
            Mat batchX = x(roi);
            roi = Rect(randomNum, 0, batch, y.rows);
            Mat batchY = y(roi);

            Cost(batchX, batchY, lambda);
            cout<<"learning step: "<<converge
				<<", Cost function value = "<<cost
				<<", randomNum = "<<randomNum
				<<endl;
            if(fabs((cost - lastcost) / cost) <= 1e-6 && converge > 0) break;
            if(cost <= 0) break;
            lastcost = cost;
            smrW -= lrate * smrWg;
            for(int i=0; i<nLayers; i++){
                scW[i] -= lrate * scWg[i];
                //HiddenLayers[i].W2 -= lrate * HiddenLayers[i].W2grad;
                scb[i] -= lrate * scbg[i];
                //HiddenLayers[i].b2 -= lrate * HiddenLayers[i].b2grad;
            }
            ++ converge;
        }
    }
}

cv::Mat StackedNetwork::resultProdict(Mat &x){

	vector<Mat> acti;
    acti.push_back(x);
    for(int i=1; i <= nLayers; i++){

        Mat tmpacti = scW[i-1] * acti[i - 1] + repeat(scb[i - 1], 1, x.cols);
        acti.push_back(sigmoid(tmpacti));
    }
    Mat M = smrW * acti[acti.size() - 1];
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M = M - repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);
    log(p, tmp);
    //cout<<tmp.t()<<endl;
    Mat result = Mat::ones(1, tmp.cols, CV_64FC1);
    for(int i=0; i<tmp.cols; i++){

        double maxele = tmp.ATD(0, i);
        int which = 0;
        for(int j=1; j<tmp.rows; j++){

            if(tmp.ATD(j, i) > maxele){
                
				maxele = tmp.ATD(j, i);
                which = j;
            }
        }
        result.ATD(0, i) = which;
    }
    acti.clear();
    return result;
}
