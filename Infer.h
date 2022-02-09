#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <inference_engine.hpp>

#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Detector
{
public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    Detector();
    ~Detector();

    bool init(string xml_path, double cof_threshold, double nms_area_threshold, string device);

    bool uninit();

    bool process_frame(Mat& inframe, vector<Object> &detected_objects, int imsize);

private:
    double sigmoid(double x);

    vector<int> get_anchors(int net_grid);

    bool parse_yolov5(const Blob::Ptr &blob, int net_grid, float cof_threshold, vector<Rect>& o_rect, vector<float>& o_rect_cof, int imsize);

    Rect detet2origin(const Rect& dete_rect, float rate_to, int top, int left);

    ExecutableNetwork _network;

    OutputsDataMap _outputinfo;

    string _input_name;

    string _xml_path;

    double _cof_threshold;

    double _nms_area_threshold;
};
#endif