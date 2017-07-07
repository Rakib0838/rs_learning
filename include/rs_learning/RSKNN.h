
#ifndef RSKNN_HEADER
#define RSKNN_HEADER


#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <rs_learning/RSClassifier.h>


#include <uima/api.hpp>
#include <rs/scene_cas.h>
//RS
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>

class RSKNN : public RSClassifier
{

public:
    RSKNN();

    void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

    void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name,std::string obj_classInDouble);

    void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det);

    void RsAnnotation (uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode);

    void classifyKNN(std::string train_matrix_name,std::string train_label_name,
                   std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble);

    void classifyOnLiveDataKNN(std::string train_matrix_name, std::string train_label_name, cv::Mat test_mat, double &det);

   void processVFHFeatureKNN(std::string train_matrix_name,std::string train_label_name,std::string set_mode, std::string dataset_use,std::string feature_use,
         std::vector<rs::Cluster> clusters, RSKNN *obj_VFH, cv::Mat &color,std::vector<std::string> models_label, uima::CAS &tcas);

   void  processCaffeFeatureKNN(std::string train_matrix_name,std::string train_label_name,
          std::string set_mode, std::string dataset_use,std::string feature_use, std::vector<rs::Cluster> clusters,
           RSKNN *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas);



    ~ RSKNN();
};

#endif
