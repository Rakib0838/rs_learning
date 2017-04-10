
#ifndef RSSVM_HEADER
#define RSSVM_HEADER


#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <rs_learning/RSClassifier.h>


class RSSVM : public RSClassifier , public CvSVM
{

public:

    RSSVM();
    void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);
    void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble);
    ~ RSSVM();
};

#endif
