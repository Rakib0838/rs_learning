
#ifndef RSCLASSIFIER_HEADER
#define RSCLASSIFIER_HEADER


#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/ml/ml.hpp>



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
 
 void getLabels(const std::string path,  std::map<std::string, double> &input_file);
 
 void readDescriptorAndLabel(std::string matrix_name, std::string label_name, cv::Mat &des_matrix, cv::Mat &des_label);
  
 std::string saveOrLoadTrained(std::string trained_file_name);
  
void evaluation(std::vector<int> test_label, std::vector<int> predicted_label,std::string obj_classInDouble);
  
~ RSClassifier();

};

#endif
