#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Basefun.h"

#define ATD at<double>

using namespace std;
using namespace cv;

Mat concatenateMat(vector<Mat> &vec){

	int height = vec[0].rows;
	int width = vec[0].cols;
	Mat res = Mat::zeros(height*width, vec.size(), CV_64FC1);
	// for debugging...
	// cout << res(Range(300,350), Range(0,1)) << endl;
	for(int i=0; i < vec.size(); i++){
		Mat img(height, width, CV_64FC1);

		vec[i].convertTo(img, CV_64FC1);
		// ptmat is column vector
		Mat ptmat = img.reshape(0, height*width); 
		// if (i == 0) cout << ptmat(Range(300,350), Range(0,1)) << endl;
		Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
		Mat subView = res(roi);
		ptmat.copyTo(subView);
	}
	// for debugging...
	// cout << res(Range(300,350),Range(0,1)) << endl;
	divide(res, 255.0, res);
	return res;
}

int ReverseInt(int i){

	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<Mat> &vec){

	ifstream file(filename, ios::binary);
	// for debugging...
	// cout << "file status: " << file.is_open() << endl;
	if(file.is_open()){
		
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		// for debugging...
		// cout << "number_of_images: " << number_of_images << endl;
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for(int i = 0; i < number_of_images; i++){
		
			// for debugging...
			if ((i+1) % 5000 == 0){
			
				cout << "finish reading no." << (i - 4998) << "~" << (i + 1) << "  images" << endl;
			}
			
			Mat tpmat = Mat::zeros(n_rows, n_cols, CV_8UC1);
			for(int r=0; r<n_rows; r++){
				for(int c=0; c<n_cols; c++){
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					tpmat.at<uchar>(r,c) = (int) temp;
				}
			}
			vec.push_back(tpmat);
		}
	}
}

void read_Mnist_Label(string filename, Mat &mat){

	ifstream file(filename, ios::binary);
	if(file.is_open()){
	
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		
		for(int i=0; i < number_of_images; ++i){

			// for debugging...
			if ((i + 1) % 5000 == 0){
			
				cout << "finish reading no." << (i - 4998) << "~" << (i + 1) << " labels" << endl;
			}
		
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			mat.ATD(0,i) = (double)temp;	
		}
	}
}

Mat sigmoid(Mat &M){

	Mat temp;
	exp(-M, temp);
	return 1.0 / (temp+1.0);
}

Mat dsigmoid(Mat &a){

	Mat res = 1.0 - a;
	res = res.mul(a);
	return res;
}

void readData(Mat &x, Mat &y, string xpath, string ypath, int number_of_images){

	//read MNIST iamge into OpenCV Mat vector
	int image_size = 28 * 28;
	vector<Mat> vec;
	//vec.resize(number_of_images, cv::Mat(28, 28, CV_8UC1));
	read_Mnist(xpath, vec);
	//read MNIST label into double vector
	y = Mat::zeros(1, number_of_images, CV_64FC1);
	read_Mnist_Label(ypath, y);
	x = concatenateMat(vec);
}