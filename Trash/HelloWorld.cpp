#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <inference_engine.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

int main(){
	std::cout << "Hello world!" << std::endl;

	//create a gui window:
    namedWindow("Output",1);
    
    //initialize a 120X350 matrix of black pixels:
    Mat output = Mat::zeros( 120, 350, CV_8UC3 );
    
    //write text on the matrix:
    putText(output, "Hello World :)", cv::Point(15,70), FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,255,0), 4);
    
    //display the image:
    imshow("Output", output);
    
    //wait for the user to press any key:
    waitKey(0);

    std::cout << "Start IE" << std::endl;
    Core ie;
    std::cout << "Success IE" << std::endl;
    waitKey(0);

	return 0;
}

//Компиляция:  g++ -I/usr/local/include/opencv4 HelloWorld.cpp -o HelloWorld -L/usr/local/lib -llibopencv_core -llibopencv_imgproc -llibopencv_imgcodecs -llibopencv_highgui


//Its work!
//Comp: g++ -I/usr/local/include/opencv4 HelloWorld.cpp -o HelloWorld -L/usr/local/lib -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_img_hash -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann -lopencv_face -lopencv_photo -lopencv_xphoto
