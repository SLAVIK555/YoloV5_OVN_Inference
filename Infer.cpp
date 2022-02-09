#include "Infer.h"

Detector::Detector(){}

Detector::~Detector(){}

bool Detector::parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof, int imsize){
    vector<int> anchors = get_anchors(net_grid);

    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();

    const float *output_blob = blobMapped.as<float *>();

    int item_size = 8;
    size_t anchor_n = 3;


    for(int n=0; n<anchor_n; ++n)
        for(int i=0; i<net_grid; ++i)
            for(int j=0; j<net_grid; ++j)
            {
                //std::cout << "Ok1" << std::endl;
                double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4];
                box_prob = sigmoid(box_prob);
                //std::cout << "Ok2" << std::endl;

                if(box_prob < cof_threshold){
                    continue;
                    //std::cout << "!Ok1" << std::endl;
                }

                //std::cout << "Ok3" << std::endl;
                
                double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 3];
                //std::cout << "Ok4" << std::endl;
               
                double max_prob = 0;
                int idx=0;
                //std::cout << "Ok5" << std::endl;

                for(int t=5;t<8;++t){
                    //std::cout << "t: " << t << std::endl;
                    double tp = output_blob[(n*net_grid*net_grid*item_size) + (i*net_grid*item_size) + (j*item_size) + t];
                    //std::cout << "tp: " << tp << std::endl;
                    tp = sigmoid(tp);
                    //std::cout << "tp: " << tp << std::endl;
                    //std::cout << "max_prob: " << max_prob << std::endl;
                    //std::cout << "idx: " << idx << std::endl;
                    if(tp > max_prob){
                        max_prob = tp;
                        idx = t;
                    }
                    //std::cout << "max_prob: " << max_prob << std::endl;
                    //std::cout << "idx: " << idx << std::endl;
                    //std::cout << "_________________" << std::endl;
                }
                //std::cout << "Ok6" << std::endl;

                float cof = box_prob * max_prob; 

                if(cof < cof_threshold){
                    continue;
                    //std::cout << "!Ok2" << std::endl;
                }

                //std::cout << "Ok7" << std::endl;

                x = (sigmoid(x)*2 - 0.5 + j)*imsize/net_grid;
                y = (sigmoid(y)*2 - 0.5 + i)*imsize/net_grid;
                w = pow(sigmoid(w)*2,2) * anchors[n*2];
                h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];

                double r_x = x - w/2;
                double r_y = y - h/2;
                //std::cout << "Ok8" << std::endl;

                Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                //std::cout << "Ok9" << std::endl;
            }


    if(o_rect.size() == 0){
        return false;
    }
    else{
        return true;
    }
}


bool Detector::init(string xml_path, double cof_threshold, double nms_area_threshold, string device="GPU"){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;

    std::cout << "Initialize IE Core" << std::endl;
    Core ie;

    std::cout << "Reading network" << std::endl;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 

    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    InputInfo::Ptr& input = inputInfo.begin()->second;

    _input_name = inputInfo.begin()->first;

    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();

    SizeVector& inSizeVector = inputShapes.begin()->second;

    cnnNetwork.reshape(inputShapes);

    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());

    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }

    _network =  ie.LoadNetwork(cnnNetwork, device);

    return true;
}


bool Detector::uninit(){
    return true;
}


bool Detector::process_frame(Mat& inframe, vector<Object>& detected_objects, int imsize){

    if(inframe.empty()){
        std::cout << "input frame is empty" << std::endl;
        return false;
    }

    resize(inframe,inframe,Size(imsize,imsize));

    cvtColor(inframe,inframe,COLOR_BGR2RGB);

    size_t img_size = imsize*imsize;

    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();

    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);

    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();

    float* blob_data = blobMapped.as<float*>();

    //convert to nchw (Number of batch(1), Channel, Height, Width)
    for(size_t row =0;row<imsize;row++){
        for(size_t col=0;col<imsize;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*imsize + col] = float(inframe.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }

    infer_request->Infer();

    vector<Rect> origin_rect;

    vector<float> origin_rect_cof;

    int s[3] = {static_cast<int>(imsize/8),static_cast<int>(imsize/16),static_cast<int>(imsize/32)};

    int i=0;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        parse_yolov5(blob, s[i], _cof_threshold, origin_rect, origin_rect_cof, imsize);
        ++i;
    }


    vector<int> final_id;

    dnn::NMSBoxes(origin_rect, origin_rect_cof, _cof_threshold, _nms_area_threshold, final_id);

    for(int i=0;i<final_id.size();++i){
        Rect resize_rect= origin_rect[final_id[i]];
        detected_objects.push_back(Object{origin_rect_cof[final_id[i]],"",resize_rect});
    }

    return true;
}


double Detector::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

vector<int> Detector::get_anchors(int net_grid){
    vector<int> anchors(6);
    int a40[6] = {10,13, 16,30, 33,23};
    int a20[6] = {30,61, 62,45, 59,119};
    int a10[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    else if(net_grid == 10){
        anchors.insert(anchors.begin(),a10,a10 + 6);
    }
    return anchors;
}























// #include <iostream>

// #include "opencv2/opencv.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgcodecs/imgcodecs.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

// #include <inference_engine.hpp>
// #include <ngraph/ngraph.hpp>

// using namespace std;
// using namespace cv;
// using namespace InferenceEngine;

// xmlModel = "/home/slava/YoloV5Infer/FCB_IR_Models/L_model/best_FP16.xml"
// binModel = "/home/slava/YoloV5Infer/FCB_IR_Models/L_model/best_FP16.bin"

// int main(){
// 	std::cout << "YoloV5 OpenVINO Inference!" << std::endl;

//     // --------------------------- Step 1. Initialize inference engine core
//     std::cout << "Loading Inference Engine" << std::endl;
//     Core ie;

//     //Initialization of reasoning engine
//     std::cout << "Reading CNN Network" << std::endl;
//     //Read in the xml file, which will automatically read the corresponding bin file in the directory of the xml file, without having to specify it manually
//     auto cnnNetwork = ie.ReadNetwork(xmlModel); 

//     //Getting formatting information for input data from a model
//     InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

//     InputInfo::Ptr& input = inputInfo.begin()->second;
    
//     _input_name = inputInfo.begin()->first;
//     input->setPrecision(Precision::FP32);
//     input->getInputData()->setLayout(Layout::NCHW);
//     ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
//     SizeVector& inSizeVector = inputShapes.begin()->second;
//     cnnNetwork.reshape(inputShapes);
//     //Format for deriving inferences from models
//     _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
//     for (auto &output : _outputinfo) {
//         output.second->setPrecision(Precision::FP32);
//     }
// //Get an executable network, where the CPU is the inferred device, and optionally the GPU, where the GPU is the core display inside the intel chip
// //Configuring the GPU running environment required for verification greatly improves the speed of inference using GPU mode. The way of configuring GPU environment will be mentioned after taking the CPU deployment first.
// _network =  ie.LoadNetwork(cnnNetwork, "CPU");

// 	return 0;
// }

// //Компиляция:  g++ -I/usr/local/include/opencv4 HelloWorld.cpp -o HelloWorld -L/usr/local/lib -llibopencv_core -llibopencv_imgproc -llibopencv_imgcodecs -llibopencv_highgui


// //Its work!
// //Makefile