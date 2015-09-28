#ifndef BASEFUN_H
#define BASEFUN_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>

cv::Mat concatenateMat(std::vector<cv::Mat> &vec);

int ReverseInt(int i);

void read_Mnist(std::string filename, std::vector<cv::Mat> &vec);

void read_Mnist_Label(std::string filename, cv::Mat &mat);

cv::Mat sigmoid(cv::Mat &mat);

cv::Mat dsigmoid(cv::Mat &mat);

void readData(cv::Mat &x, cv::Mat &y, std::string xpath, std::string ypath, int number_of_images);

#endif