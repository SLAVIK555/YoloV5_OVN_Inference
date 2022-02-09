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
#include <ctime>

#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

//cv::Mat image;

string names[] = {"Face", "Cup", "Bottle"};

//#define IMG_640
//#define IMG_416
#define IMG_320

// static void loadimg(const char * imagename, int width, int height)
// {
// 	image = cv::imread(imagename);
// 	cout << "load image: " << imagename << " resize: w=" << width << " h=" << height << endl;
// 	cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
// }


double sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}


vector<int> get_anchors(int net_grid) {
	vector<int> anchors(6);

	int a80[6] = { 10,13, 16,30, 33,23 };
	int a40[6] = { 30,61, 62,45, 59,119 };
	int a20[6] = { 116,90, 156,198, 373,326 };

	#ifdef IMG_640
		if (net_grid == 80) {
			anchors.insert(anchors.begin(), a80, a80 + 6);
		}
		else if (net_grid == 40) {
			anchors.insert(anchors.begin(), a40, a40 + 6);
		}
		else if (net_grid == 20) {
			anchors.insert(anchors.begin(), a20, a20 + 6);
		}
	#endif

	#ifdef IMG_416
		if (net_grid == 52) {
			anchors.insert(anchors.begin(), a80, a80 + 6);
		}
		else if (net_grid == 26) {
			anchors.insert(anchors.begin(), a40, a40 + 6);
		}
		else if (net_grid == 13) {
			anchors.insert(anchors.begin(), a20, a20 + 6);
		}
	#endif

	#ifdef IMG_320
		if (net_grid == 40) {
			anchors.insert(anchors.begin(), a80, a80 + 6);
		}
		else if (net_grid == 20) {
			anchors.insert(anchors.begin(), a40, a40 + 6);
		}
		else if (net_grid == 10) {
			anchors.insert(anchors.begin(), a20, a20 + 6);
		}
	#endif

	return anchors;
}


bool parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold, vector<cv::Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& o_label) {
	vector<int> anchors = get_anchors(net_grid);

	LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();

	const float *output_blob = blobMapped.as<float *>();//float

	// int k = 0;
	// while (true){
	// 	cout << "k: " << k << endl;
	// 	cout << "ob: " << output_blob[k] << endl;
	// 	k++;
	// }


	int item_size = 8;//cx, cy, w, h, conf + number of class
	//int item_size = 1;  //make item_size useless
	size_t anchor_n = 3;

	for (int n = 0; n < anchor_n; ++n)
	{
		for (int i = 0; i < net_grid; ++i)
		{
			for (int j = 0; j < net_grid; ++j)
			{
				//if (i == 9 || j == 9)
				//{
				//	n = n;
				//}

				//double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid + 4 + j*net_grid*net_grid];//swap 4 and j
				double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 4];
				box_prob = sigmoid(box_prob);

				if (box_prob < cof_threshold)
				{
					continue;
				}

				//cout << "preOk" << endl;

				//cout << "n= " << n << " " << i << " " << j << " conf=" << box_prob << "----------------------------------------------" << endl;
				
				// double x = output_blob[n*net_grid*net_grid*item_size + i * net_grid + 0  + j * net_grid*net_grid];//swap number and j also
				// double y = output_blob[n*net_grid*net_grid*item_size + i * net_grid + 1  + j * net_grid*net_grid];
				// double w = output_blob[n*net_grid*net_grid*item_size + i * net_grid + 2  + j * net_grid*net_grid];
				// double h = output_blob[n*net_grid*net_grid*item_size + i * net_grid + 3  + j * net_grid*net_grid];
				double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 3];
				//cout << "Ok" << endl;

				double max_prob = 0;
				int idx = 0;
				for (int t = 5; t < 8; ++t) //85
				{
					//cout << "t: " << t << endl;

					//cout << "n: " << n << endl;
					//cout << "net_grid: " << net_grid << endl;
					//cout << "item_size: " << item_size << endl;
					//cout << "i: " << i << endl;
					//cout << "j: " << j << endl;
					//cout << "n*net_grid*net_grid*item_size + i * net_grid + j  + t * net_grid*net_grid: " << n*net_grid*net_grid*item_size + i * net_grid + j  + t * net_grid*net_grid << endl;
					//cout << "size:" << sizeof(output_blob)/sizeof(double); << endl;

					//double tp = output_blob[n*net_grid*net_grid*item_size + i* net_grid + j  + t * net_grid*net_grid];
					double tp = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + t];
					//cout << "tp: " << tp << endl;

					tp = sigmoid(tp);
					//cout << "tp: " << tp << endl;
					//cout << "max_prob: " << max_prob << endl;

					if (tp > max_prob) 
					{
						max_prob = tp;
						idx = t;
					}
					//cout << "idx: " << idx << endl;
					//cout << "max_prob: " << max_prob << endl;
					//cout << "------------------------" << endl;

				}


				float cof = box_prob * max_prob;
				if (cof < cof_threshold)
				{
					continue;
				}

				#ifdef IMG_640
				x = (sigmoid(x) * 2 - 0.5 + j)*640.0f / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)*640.0f / net_grid;
				#endif

				#ifdef IMG_416
				x = (sigmoid(x) * 2 - 0.5 + j)*416.0f / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)*416.0f / net_grid;
				#endif

				#ifdef IMG_320
				x = (sigmoid(x) * 2 - 0.5 + j)*320.0f / net_grid;
				y = (sigmoid(y) * 2 - 0.5 + i)*320.0f / net_grid;
				#endif

				w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
				h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

				double r_x = x - w / 2;
				double r_y = y - h / 2;

				cv::Rect rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));

				o_rect.push_back(rect);
				o_rect_cof.push_back(cof);
				o_label.push_back(idx-5);
			}
		}
	}

	if (o_rect.size() == 0) return false;
	else return true;
}

int main(){
	try{
		//string device = "CPU";
		string device = "GPU";
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/TutConv/yolov5s_my.xml";//Its didn't work, seg_fault
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/TutConv/yolov5s_my.onnx";//Its work, but very bad!
		//string model = "/home/slava/Загрузки/yolov5_cpp_openvino/demo/res/his/yolov5s.xml";//Its work, not bad
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/TutConv/Pre/yolov5s.onnx";//Its work, not bad
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/TutConv/Pre/yolov5s.xml";//Its work, not bad in case FP32 sometines means seg_fault
		//string model = "/home/slava/yolov5/runs/train/exp2/weights/best.onnx";//Its work, but very bad!
		//string model = "/home/slava/yolov5/runs/train/exp2/weights/best_openvino_model/best.xml";//Its didn't work, seg_fault
		//string model = "/home/slava/yolov5/runs/train/exp2/weights/ovn_dev_converted/best.xml";//Its work better, than abs bad
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/New_S/best.onnx";//Its work, but very bad!
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/New_S/best_openvino_model/best.xml";//Its work, but very bad!
		//string model = "/home/slava/YoloV5CppInfer/FCB_IR_Models/New_S/ovn_dev_converted/best.xml";//Its work better, than abs bad
		//string model = "/home/slava/Source/YoloV5_OVN_Inference/New_S/best.onnx";
		//string model = "/home/slava/Source/YoloV5_OVN_Inference/New_S/ovn_dev_converted/best.xml";//Its work, but very bad!
		string model = "/home/slava/Source/YoloV5_OVN_Inference/NewUbuntuConv/ColabSModel/best.xml";
		//string input_image = "face3.jpg";
		string input_video = "outpy.avi";

		cout << "Model name = " << model << endl;
		//cout << "Image name = " << input_image << endl;
		cout << "starting" << endl;
		// const Version *IEversion;
		// IEversion = GetInferenceEngineVersion();
		// cout << "InferenceEngine: API version " << IEversion->apiVersion.major << "." << IEversion->apiVersion.minor << endl;
		// cout << "InferenceEngine: Build : " << IEversion->buildNumber << endl << endl;

		// --------------------------- 1. Load inference engine -------------------------------------
		cout << "Creating Inference Engine" << endl;

		Core ie;
		// ------------------------------------------------------------------------------------------


		// --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
		cout << "Loading network files" << endl;

		/** Read network model **/
		CNNNetwork network = ie.ReadNetwork(model);
		cout << "network layer count: " << network.layerCount() << endl;
		// -----------------------------------------------------------------------------------------------------


		// --------------------------- 3. Configure input & output ---------------------------------------------

		// --------------------------- Prepare input blobs -----------------------------------------------------
		cout << "Preparing input blobs" << endl;

		// Taking information about all topology inputs
		InputsDataMap inputInfo(network.getInputsInfo());
		if (inputInfo.size() != 1){
			throw std::logic_error("Sample supports topologies with 1 input only");
		}

		auto inputInfoItem = *inputInfo.begin();

		inputInfoItem.second->setPrecision(Precision::U8);//U8 or FP16 or FP32 original is U8
		inputInfoItem.second->setLayout(Layout::NCHW);

		network.setBatchSize(1);
		size_t batchSize = network.getBatchSize();
		cout << "Batch size is " << std::to_string(batchSize) << endl;
		// -----------------------------------------------------------------------------------------------------


		// --------------------------- 4. Loading model to the device ------------------------------------------
		cout << "Loading model to the device: " << device << endl;
		ExecutableNetwork executable_network = ie.LoadNetwork(network, device);
		// -----------------------------------------------------------------------------------------------------


		// --------------------------- 5. Create infer request -------------------------------------------------
		cout << "Create infer request" << endl;
		InferRequest inferRequest_regular = executable_network.CreateInferRequest();
		// -----------------------------------------------------------------------------------------------------


		//loadimg(input_image.c_str(), inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]);

		VideoWriter video("V5outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));

		//open the video file for reading
		VideoCapture cap(input_video); 

		// if not success, exit program
		if (cap.isOpened() == false)  
		{
			cout << "Cannot open the video file" << endl;
			cin.get(); //wait for any key press
			return -1;
		}

		//int fc = 0;
		
		//clock_t start = std::clock();

		while (true)
		{
			//fc++;
			clock_t start = std::clock();

			cv::Mat image;
			bool bSuccess = cap.read(image); // read a new frame from video 
			cv::resize(image, image, cv::Size(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]), 0, 0, cv::INTER_AREA);

			//Breaking the while loop at the end of the video
			if (bSuccess == false) 
			{
				cout << "Found the end of the video" << endl;
				break;
			}

			// --------------------------- 6. Prepare input --------------------------------------------------------
			for (auto & item : inputInfo) 
			{
				Blob::Ptr inputBlob = inferRequest_regular.GetBlob(item.first);

				SizeVector dims = inputBlob->getTensorDesc().getDims();

				// Fill input tensor with images. First b channel, then g and r channels
				size_t num_channels = dims[1];
				//std::cout << "num_channles = " << num_channels << std::endl;
				size_t image_size = dims[3] * dims[2];

				MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
				if (!minput)
				{
					cout << "We expect MemoryBlob from inferRequest_regular, but in fact we were not able to cast inputBlob to MemoryBlob" << endl;
					return 1;
				}

				// locked memory holder should be alive all time while access to its buffer happens
				auto minputHolder = minput->wmap();

				auto data = minputHolder.as<PrecisionTrait<Precision::U8>::value_type *>();//U8 or FP16 or FP32 original is U8
				unsigned char* pixels = (unsigned char*)(image.data);

				//cout << "image_size = " << image_size << endl;
				// Iterate over all pixel in image (b,g,r)
				for (size_t pid = 0; pid < image_size; pid++) 
				{
					// Iterate over all channels
					for (size_t ch = 0; ch < num_channels; ++ch) 
					{
						data[ch * image_size + pid] = pixels[pid*num_channels + ch];
					}
				}
			}
			// -----------------------------------------------------------------------------------------------------

			// --------------------------- 7. Do inference ---------------------------------------------------------
			// Start sync request
			//cout << "Start inference " << endl;

			//milliseconds start_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
			inferRequest_regular.Infer();
			//milliseconds end_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

			//std::cout << "total cost time: " << (end_ms - start_ms).count() << " ms" << std::endl;
			//float total_time = (end_ms - start_ms).count() / 1000.0;
			//std::cout << "FPS: " << (float)1.0 / total_time << std::endl;
			// -----------------------------------------------------------------------------------------------------

			// --------------------------- 8. Process output -------------------------------------------------------
			//cout << "Processing output blobs" << endl;
			OutputsDataMap outputInfo(network.getOutputsInfo());

			vector<cv::Rect> origin_rect;
			vector<float> origin_rect_cof;
			vector<int> label;

			double _cof_threshold = 0.98;                
			double _nms_area_threshold = 0.9;  

			#ifdef IMG_320
				int s[3] = { 40,20,10 };
			#endif

			#ifdef IMG_416
				int s[3] = { 52,26,13 };
			#endif

			#ifdef IMG_640
				int s[3] = { 80,40,20 };
			#endif

			int i = 0;
			for (auto &output : outputInfo) 
			{
				auto output_name = output.first;
				//cout << " ------ output_name = " << output_name << endl;

				Blob::Ptr blob = inferRequest_regular.GetBlob(output_name);

				parse_yolov5(blob, s[i], _cof_threshold, origin_rect, origin_rect_cof, label);

				//cout << "label.size() = " << label.size() << endl;
				++i;
			}

			vector<int> final_id;
			cv::dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);
			//cout << "final_id.size() = " << final_id.size() << endl;

			for (int i = 0; i < final_id.size(); ++i) {
				cv::Rect resize_rect = origin_rect[final_id[i]];
				float cof = origin_rect_cof[final_id[i]];

				int xmin = resize_rect.x;
				int ymin = resize_rect.y;
				int width = resize_rect.width;
				int height = resize_rect.height;
				cv::Rect rect(xmin, ymin, width, height);
				//cout << xmin << " " << ymin << " " << width << " " << height << " "<< endl;
				//cv::putText(jpg, "label="+std::to_string(label[final_id[i]]), cv::Point2f(xmin, ymin), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 0, 0, 255 });
				//double fps = fc/(std::clock()-start);
				//double dur = std::clock() - start;
				clock_t end = std::clock();
				float fps = 1000000 / (end - start);


				cout << fps << endl;

				//cv::putText(image, to_string(fps), cv::Point2f(15, 15), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 0, 0, 0});
				cv::putText(image, "fps: " + to_string(fps), cv::Point2f(15, 15), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 0, 0, 0});
				cv::putText(image, names[label[final_id[i]]] + ", " + to_string(round(cof*100)/100), cv::Point2f(xmin, ymin), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 0, 0, 255});
				cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);

			}

			video.write(image);

			imshow("result",image);
			//waitKey(0);
			// -----------------------------------------------------------------------------------------------------


			if (waitKey(10) == 27)
			{
				cout << "Esc key is pressed by user. Stoppig the video" << endl;
				break;
			}
		}

		// if (image.data == NULL)
		// {
		// 	cout << "Valid input images were not found!" << endl;
		// }

		// Setting batch size to 1
		


		
	}


	catch (const std::exception& error) {
		cout << error.what() << endl;
		return 1;
	}


	catch (...) {
		cout << "Unknown/internal exception happened." << endl;
		return 1;
	}


	cout << "Execution successful" << endl;
	return 0;
}