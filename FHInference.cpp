#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

cv::Mat src = cv::imread("face1.jpg");

string names[] = {"Face", "Cup", "Bottle"};

string exec_device = "CPU";

//#define IMG_640
//#define IMG_416
//#define IMG_320

double sigmoid_function(double x) {
	return (1 / (1 + exp(-x)));
}

vector<int> get_anchors(int net_grid) {
	vector<int> anchors(6);

	int a80[6] = { 10,13, 16,30, 33,23 };
	int a40[6] = { 30,61, 62,45, 59,119 };
	int a20[6] = { 116,90, 156,198, 373,326 };

	if (net_grid == 40) {
		anchors.insert(anchors.begin(), a80, a80 + 6);
	}
	else if (net_grid == 20) {
		anchors.insert(anchors.begin(), a40, a40 + 6);
	}
	else if (net_grid == 10) {
		anchors.insert(anchors.begin(), a20, a20 + 6);
	}

	return anchors;
}

bool parce_yolov5(const Blob::Ptr &output_blob, int side_hw, vector<cv::Rect>& boxes, vector<float>& confidences, vector<int>& classIds){
	vector<int> anchors = get_anchors(side_hw);
	int out_c = 3;
	int side_square = side_hw*side_hw;
	int item_size = 8;//cx, cy, w, h, conf + number of class

	for (int i = 0; i <side_square; ++i) {
    	for (int c = 0; c <out_c; c++) {
    	    int row = i/side_hw;
    	    int col = i%side_hw;
    	    int object_index = c*side_square*item_size + row*side_hw + col*side_square;

    	    //Threshold filtering
    	    float conf = sigmoid_function(output_blob[object_index + 4]);
        	if (conf <0.25) {
        	    continue;
        	}

        	int anchor_index = 0;
        	float stride = 320/side_hw;
        	//parse cx, cy, width, height
        	float x = (sigmoid_function(output_blob[object_index]) * 2-0.5 + col)*stride;
        	float y = (sigmoid_function(output_blob[object_index + 1]) * 2-0.5 + row)*stride;
        	float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
        	float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
        	float max_prob = -1;
        	int class_index = -1;

        	//parse category
        	for (int d = 5; d <item_size; d++) {
        	    float prob = sigmoid_function(output_blob[object_index + d]);
        	    if (prob> max_prob) {
        	        max_prob = prob;
        	        class_index = d-5;
        	    }
        	}

        	// //Convert to top-left, bottom-right coordinates
        	// int x1 = saturate_cast<int>((x-w/2) * scale_x);//top left x
        	// int y1 = saturate_cast<int>((y-h/2) * scale_y);//top left y
        	// int x2 = saturate_cast<int>((x + w/2) * scale_x);//bottom right x
        	// int y2 = saturate_cast<int>((y + h/2) * scale_y);//bottom right y

        	double r_x = x - w / 2;
			double r_y = y - h / 2;
			cv::Rect rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));

        	//parse the output
        	classIds.push_back(class_index);
        	confidences.push_back((float)conf);
        	boxes.push_back(rect);
        	//rectangle(src, Rect(x1, y1, x2-x1, y2-y1), Scalar(255, 0, 255), 2, 8, 0);
    	}
	}
}


int main(){
	//Create IE plug-in, query supporting hardware devices
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i <availableDevices.size(); i++) {
    printf("supported device name: %s/n", availableDevices[i].c_str());
	}

	//Load the detection model
	//auto network = ie.ReadNetwork("/home/slava/yolov5/runs/train/exp2/weights/best_openvino_model/best.xml", "/home/slava/yolov5/runs/train/exp2/weights/best_openvino_model/best.bin");
	//auto network = ie.ReadNetwork("/home/slava/yolov5/runs/train/exp2/weights/ovn_dev_converted/best.xml", "/home/slava/yolov5/runs/train/exp2/weights/ovn_dev_converted/best.bin");
	auto network = ie.ReadNetwork("/home/slava/yolov5/runs/train/exp2/weights/best.onnx");
	InputsDataMap inputInfo(network.getInputsInfo());


	//Set the input format
	for (auto &item: input_info) {
	    auto input_data = item.second;
	    input_data->setPrecision(Precision::FP32);
	    input_data->setLayout(Layout::NCHW);
	    input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
	    input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}

	//Set the output format
	for (auto &item: output_info) {
	    auto output_data = item.second;
	    output_data->setPrecision(Precision::FP32);
	}
	auto executable_network = ie.LoadNetwork(network, exec_device);
	InferRequest infer_request = executable_network.CreateInferRequest();


	int64 start = getTickCount();
	/** Iterating over all input blobs **/
	for (auto & item: input_info) {
	    auto input_name = item.first;

	    /** Getting input blob **/
	    auto input = infer_request.GetBlob(input_name);
	    size_t num_channels = input->getTensorDesc().getDims()[1];
	    size_t h = input->getTensorDesc().getDims()[2];
	    size_t w = input->getTensorDesc().getDims()[3];
	    size_t image_size = h*w;
	    cv::Mat blob_image;
	    resize(src, blob_image, Size(w, h));//##src
	    cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

	   //NCHW
	    float* data = static_cast<float*>(input->buffer());
	    for (size_t row = 0; row <h; row++) {
	        for (size_t col = 0; col <w; col++) {
	            for (size_t ch = 0; ch <num_channels; ch++) {
	                data[image_size*ch + row*w + col] = float(blob_image.at<Vec3b>(row, col)[ch])/255.0;
	            }
	        }
	    }
	}

	//Perform prediction
	infer_request.Infer();


	vector<cv::Rect> origin_rect;
	vector<float> origin_rect_cof;
	vector<int> label;
	int s[3] = { 40,20,10 };

	int i = 0;
	for (auto &output : outputInfo) 
	{
		auto output_name = output.first;
		cout << " ------ output_name = " << output_name << endl;
		Blob::Ptr blob = inferRequest_regular.GetBlob(output_name);

		parse_yolov5(blob, s[i], origin_rect, origin_rect_cof, label);

		cout << "label.size() = " << label.size() << endl;
		++i;
	}


	vector<int> indices;
	cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, 0.25, 0.5, indices);
	for (size_t i = 0; i <indices.size(); ++i)
	{
	    int idx = indices[i];
	    Rect box = boxes[idx];
	    cv::rectangle(src, box, Scalar(140, 199, 0), 4, 8, 0);
	}

	//vector<int> anchors = get_anchors(net_grid);


// 	for (int i = 0; i <side_square; ++i) {//## 
//     for (int c = 0; c <out_c; c++) {//##out_c = 3
//         int row = i/side_h;
//         int col = i%side_h;
//         int object_index = c*side_data_square + row*side_data_w + col*side_data;

//        //Threshold filtering
//         float conf = sigmoid_function(output_blob[object_index + 4]);
//         if (conf <0.25) {
//             continue;
//         }

//        //parse cx, cy, width, height
//         float x = (sigmoid_function(output_blob[object_index]) * 2-0.5 + col)*stride;
//         float y = (sigmoid_function(output_blob[object_index + 1]) * 2-0.5 + row)*stride;
//         float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
//         float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
//         float max_prob = -1;
//         int class_index = -1;

//        //parse category
//         for (int d = 5; d <85; d++) {
//             float prob = sigmoid_function(output_blob[object_index + d]);
//             if (prob> max_prob) {
//                 max_prob = prob;
//                 class_index = d-5;
//             }
//         }

//        //Convert to top-left, bottom-right coordinates
//         int x1 = saturate_cast<int>((x-w/2) * scale_x);//top left x
//         int y1 = saturate_cast<int>((y-h/2) * scale_y);//top left y
//         int x2 = saturate_cast<int>((x + w/2) * scale_x);//bottom right x
//         int y2 = saturate_cast<int>((y + h/2) * scale_y);//bottom right y

//        //parse the output
//         classIds.push_back(class_index);
//         confidences.push_back((float)conf);
//         boxes.push_back(Rect(x1, y1, x2-x1, y2-y1));
//        //rectangle(src, Rect(x1, y1, x2-x1, y2-y1), Scalar(255, 0, 255), 2, 8, 0);
//     }
// }
}
