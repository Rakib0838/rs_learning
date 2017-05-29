
#ifndef RSCLASSIFIER_HEADER
#define RSCLASSIFIER_HEADER


#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>
#include <rs/types/all_types.h>

using namespace rs;



class RSClassifier
{
public:

  RSClassifier();
  
 //The below function takes descriptor matrix (train_matrix_name) and its label (train_label_name),
   //which is stored in rs_learning/data folder and generates trained matrix (trained_file_name), which is found
   //stored in rs_learning/data/trainedData folder.  
  virtual void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)=0;
  
 //The function takes trained matrix and test data  matrix and its label and shows the prediction results 
  virtual void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)=0;
 
  virtual void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det)=0;

 void getLabels(const std::string path,  std::map<std::string, double> &input_file);
 
 void readDescriptorAndLabel(std::string matrix_name, std::string label_name, cv::Mat &des_matrix, cv::Mat &des_label);
  
 std::string saveOrLoadTrained(std::string trained_file_name);
  
void evaluation(std::vector<int> test_label, std::vector<int> predicted_label,std::string obj_classInDouble);

void drawCluster(cv::Mat input , cv::Rect rect, const std::string &label);

//void  processPclFeature(std::string memory_name, std::vector<Cluster> clusters, RSClassifier* po , cv::Mat &color  );
void  processVFHFeature(std::string memory_name, std::vector<Cluster> clusters, RSClassifier* po , cv::Mat &color, std::vector<std::string> models_label );

void  processCaffeFeature(std::string memory_name, std::vector<Cluster> clusters, RSClassifier* obj , cv::Mat &color, std::vector<std::string> models_label);

void setLabels(std::string file_name, std::vector<std::string> &my_annotation);

~ RSClassifier();

};

#endif
