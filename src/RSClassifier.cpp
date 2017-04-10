#include<iostream>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <rs_learning/RSClassifier.h>
#include <map>
#include <yaml-cpp/yaml.h>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

RSClassifier::RSClassifier()
{
}

// To read the class names and represent them as float numbers, which is needed to declare the
// class labels..........................................................

void RSClassifier::getLabels(const std::string path,  std::map<std::string, double> &input_file)
{
    double class_label=1;
    std::ifstream file(path.c_str());
    std::string str;
    while (std::getline(file, str))
    {

         input_file[str]=class_label;
       class_label=class_label+1;
    }
}


// To read the descriptors matrix and it's label from rs_learning/data folder...........

void RSClassifier::readDescriptorAndLabel(std::string matrix_name, std::string label_name,
                              cv::Mat &des_matrix, cv::Mat &des_label)
{

       cv::FileStorage fs;
         std::string packagePath = ros::package::getPath("rs_learning")+'/';
          std::string savePath ="data/";

          if(!boost::filesystem::exists(packagePath+savePath))
             {
              std::cout<<"Train data can be not found"<<std::endl;

             }

   fs.open(packagePath+savePath + matrix_name + ".yaml", cv::FileStorage::READ);
           fs[matrix_name]>> des_matrix;

   fs.open(packagePath+savePath + label_name + ".yaml", cv::FileStorage::READ);
            fs [label_name] >> des_label;

}

// To show the confusion matrix and accuracy result...........................

void RSClassifier::evaluation(std::vector<int> test_label, std::vector<int> predicted_label, std::string obj_classInDouble)
{

    std::map < std::string, double > object_label;

    std::string resourcePath;
      std::string lebel_path="data/"+ obj_classInDouble +".txt";

      resourcePath=ros::package::getPath("rs_learning")+'/';

      if(!boost::filesystem::exists(resourcePath+lebel_path))
         {
          std::cout<<"objects.txt file is not found"<<std::endl;

         };

    // To read the object class names from rs_resources/object_dataset/objects.txt.......

      getLabels(resourcePath+lebel_path,object_label);

    //Declare the confusion matrix which takes test data label (test_label) and predicted_label as inputs.
    // It's size is defined by the number of classes.

 std::vector <vector<int> >confusion_matrix(object_label.size(), vector<int>(object_label.size(),0));

  for(int i=0; i<test_label.size();i++){
 confusion_matrix[test_label[i]-1][predicted_label[i]-1]= confusion_matrix[test_label[i]-1][predicted_label[i]-1]+1;
        }

  //To show confusion matrix .............................................

      std::cout<<"confusion_matrix:"<<std::endl;
             for(int i=0; i<object_label.size(); i++){
                  for(int j=0; j<object_label.size();j++)
                  {
                    std::cout<< confusion_matrix[i][j]<<" ";
                  } std::cout<<std::endl;
              }
    //calculation of classifier accuracy........................................
        double c=0;
      for(int i=0;i<object_label.size();i++){
           c=c+confusion_matrix[i][i];
      }
           double Accuracy =(c/test_label.size())*100;
           std::cout<<"classifier Accuray:"<< Accuracy <<std::endl;

}


//To save or load the classifier's trained data in rs_learning/data/trainedData....

std::string RSClassifier::saveOrLoadTrained(std::string trained_file_name)
{
      std::string packagePath;
      std::string save_train="data/trainedData/";

      packagePath=ros::package::getPath("rs_learning")+'/';

      if(!boost::filesystem::exists(packagePath+save_train))
         {
          std::cout<<"Trained data can not recognize"<<std::endl;
         }
      std::string a= packagePath+ save_train +trained_file_name +".xml";

      return a;
}

RSClassifier::~ RSClassifier()
{
}
