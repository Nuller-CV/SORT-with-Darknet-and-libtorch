#include <iostream>
#include "yolo_v2_class.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <unistd.h>
#include <tracking/include/SORT.h>

#define HOME_PATH       string("/home/nuller-cv/soft/darknet/")
#define CFG_FILE        HOME_PATH + string("cfg/yolov4-tiny.cfg") // train or test configure file of darknet
#define WEIGHTS_FILE    HOME_PATH + string("yolov4-tiny.weights") // YOLO train weights file
#define LABEL_FILE      HOME_PATH + string("data/coco.names")     // label names file

#pragma comment(lib, "yolo_cpp_dll.lib") // import YOLO lib
//#pragma comment(lib, "opencv_world340d.lib") // import opencv lib

using namespace cv;
using namespace std;

vector<Scalar> color_map = {
        Scalar(255, 222, 173 ),
        Scalar(118,238,198),
        Scalar(123,104,238 ),
        Scalar(238, 99, 99 ),
        Scalar(102, 205, 0 ),
        Scalar(205,205,0 ),
        Scalar(255,165,0 ),
};

// following code from yolo_console_dll.sln
/**
 * @function:   draw the box and label or other information about detection
 * @param:      mat_img:                         the detected image
 * @param:      result_vec:                      prediction result from detector of darknet
 * @param:      obj_names:                       label names list
 * @param:      current_det_consume_per_image:   the time consumption of YOLO detection(ms)
 * @param:      current_cap_fps:                 opencv capture fps
 * @return:     void
 * */
void draw_boxes(Mat mat_img, vector<bbox_t> result_vec, vector<string> obj_names, double current_det_consume_per_image = -1, double current_cap_fps = -1, int current_gpu_id=-1)
{
    for (auto &i : result_vec)
    {
        Scalar color = obj_id_to_color(i.obj_id);
        rectangle(mat_img, Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " +   to_string(i.track_id);

            //Size const text_size = getTextSize(obj_name, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            //int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            // rectangle(mat_img, Point2f(  max((int)i.x - 1, 0),   max((int)i.y - 30, 0)),
            //              Point2f(  min((int)i.x + max_width, mat_img.cols - 1),   min((int)i.y, mat_img.rows - 1)),color, CV_FILLED, 8, 0);

            putText(mat_img, obj_name + " pro: " + to_string(i.prob*100).substr(0,2) + "%", Point2f(i.x, i.y - 10), FONT_HERSHEY_DUPLEX, 1.2, color, 2);
        }
    }
    /*if (current_det_consume_per_image >= 0 && current_cap_fps >= 0) {
        string con_str = "Detection consume per image: " +   to_string(current_det_consume_per_image).substr(0, 5) + "ms";
        putText(mat_img, con_str, Point2f(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 0), 2);
        string cap_fps="FPS capture: " +   to_string(current_cap_fps).substr(0, 5) + "  GPU id: " + to_string(current_gpu_id);
        putText(mat_img, cap_fps, Point2f(10, 45), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 0), 2);

    }*/
    putText(mat_img,"YOLO", Point2f(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(72,118,255 ), 2);
}

// by https://blog.csdn.net/weixin_42183449/article/details/80545487
Rect2d correct_box(Rect2d origin, Mat img){
    Rect2d out;

    out.x = origin.x <0 ? 1: origin.x;
    out.y = origin.y <0 ? 1: origin.y;
    out.width = origin.width;
    out.height = origin.height;

    if(origin.x + origin.width > img.cols - 1)
        out.x = img.cols - (origin.width + 1);

    if(origin.y + origin.height > img.rows - 1)
        out.y = img.rows - (origin.height + 1);

    if(origin.width > img.cols)
        origin.width = 0;

    if(origin.height > img.rows)
        origin.height = 0;
    return out;
}

void draw_track_boxe(Mat& mat_img,Track t)
{
    auto i = t.box;
    if(i.width >0 && i.height>0){
        i = correct_box(i, mat_img);
        rectangle(mat_img, Rect(i.x, i.y, i.width, i.height), color_map[t.id % color_map.size()], 2);
        putText(mat_img, to_string(t.id), Point2f(i.x, i.y), FONT_HERSHEY_DUPLEX, 1.2, Scalar(0, 0, 0), 2);
        putText(mat_img,"SORT", Point2f(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(255,118, 72), 2);
    }

}

vector<cv::Rect2f> bbox2rect(vector<bbox_t> result_vec){

    vector<cv::Rect2f> out;
    for(auto b: result_vec){
        if(b.obj_id == 0)
            out.push_back(cv::Rect2f(b.x, b.y, b.w, b.h));
    }

    return out;
}

/**
 * @function:   get label from file
 * @return:     label list [vector<string>]
 * */
vector<string> objects_names_from_file()
{
    ifstream file(LABEL_FILE);
    vector<string> file_lines;
    if (!file.is_open()) return file_lines;
    for (  string line; getline(file, line);) file_lines.push_back(line);
    cout << "object names loaded \n";
    return file_lines;
}

/**
 * @function:   detect from camera using YOLO
 * @param:      obj_names: label names list
 * */
int detect_camera(vector<string> obj_names)
{
    double fps;
    double exc_time_4_capture = 0;
    double exc_time_4_detection = 0;

    VideoCapture capture(0);
    array<int64_t, 2> orig_dim{int64_t(capture.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(capture.get(cv::CAP_PROP_FRAME_WIDTH))};
    Detector detector(CFG_FILE, WEIGHTS_FILE,0); // init detector
    SORT tracker(orig_dim);
    if(!capture.isOpened())
        return -1;

    Mat img_detect, img_track, frame, combine;

    // get camera
    while(1)
    {
        exc_time_4_capture = (double)getTickCount();

        capture>>frame;
        img_detect = frame.clone();
        img_track = frame.clone();

        exc_time_4_capture = ((double)getTickCount() - exc_time_4_capture) / getTickFrequency();

        exc_time_4_detection = (double)getTickCount();

        vector<bbox_t> result_vec = detector.detect(frame);
        auto dets = bbox2rect(result_vec);
        auto trks = tracker.update(dets);
        draw_boxes(img_detect, result_vec, obj_names, (exc_time_4_detection + exc_time_4_capture)*1000, capture.get(CAP_PROP_FPS), detector.cur_gpu_id);
        for (auto &t:trks) {
            // draw_bbox(frame, t.box, to_string(t.id), color_map(t.id));
            // draw_trajectories(frame, repo.get().at(t.id).trajectories, color_map(t.id));
            draw_track_boxe(img_track, t);
        }

        exc_time_4_detection = ((double)getTickCount() - exc_time_4_detection) / getTickFrequency();
        // draw_boxes(frame, result_vec, obj_names, (exc_time_4_detection + exc_time_4_capture)*1000, capture.get(CAP_PROP_FPS), detector.cur_gpu_id);
        hconcat(img_detect, img_track, combine);
        namedWindow("opencv & YOLO", WINDOW_NORMAL);
        imshow("YOLO & SORT", combine);
        waitKey(30);


//        cout<< "detection time:" <<to_string(exc_time_4_detection*1000).substr(0, 4);
//        cout<< "; capture time: "<< to_string(exc_time_4_capture*1000).substr(0, 4);
//        cout<< "; total time:" << to_string((exc_time_4_capture + exc_time_4_detection)*1000).substr(0, 4)<< endl;
    }

    return 0;
}



int main()
{
    // get label name list
    vector<string> obj_names = objects_names_from_file();

    detect_camera(obj_names);
    //detect_img(obj_names);
    return 0;
}