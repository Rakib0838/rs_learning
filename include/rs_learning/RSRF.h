
#ifndef RSRF_HEADER
#define RSRF_HEADER


#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <rs_learning/RSClassifier.h>

//RS
#include <uima/api.hpp>
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>

class RSRF : public RSClassifier
{

public:
    RSRF();

    void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

    void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name,std::string obj_classInDouble);

    void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det);

    void RsAnnotation (uima::CAS &tcas,std::string class_name, rs::Cluster &cluster);

    ~ RSRF();
};

#endif
