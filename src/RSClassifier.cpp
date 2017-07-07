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

#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include<rs/utils/time.h>
using namespace cv;

RSClassifier::RSClassifier()
{
}


void RSClassifier::setLabels(std::string file_name, std::vector<std::string> &my_annotation)
{
    std::string packagePath = ros::package::getPath("rs_learning");
     std::string savePath = packagePath + "/data/";

         // To check the resource path................................................
     if(!boost::filesystem::exists(savePath))
        {
         std::cout<<"folder called data is not found to read the class label from .txt file >>>"<<std::endl;
        }


    std::ifstream file((savePath+file_name+".txt").c_str());

  std::string str;
  std::vector<std::string> split_str;

  while(std::getline(file ,str))
   {
     boost::split(split_str,str,boost::is_any_of(":"));
      my_annotation.push_back(split_str[0]);

    }

}

// To read the class names and represent them as float numbers, which is needed to declare the
// class labels..........................................................

void RSClassifier::getLabels(const std::string path,  std::map<std::string, double> &input_file)
{
  double class_label = 1;
  std::ifstream file(path.c_str());
  std::string str;
  while(std::getline(file, str))
  {

    input_file[str] = class_label;
    class_label = class_label + 1;
  }
}


// To read the descriptors matrix and it's label from rs_learning/data folder...........

void RSClassifier::readDescriptorAndLabel(std::string matrix_name, std::string label_name,
    cv::Mat &des_matrix, cv::Mat &des_label)
{

  cv::FileStorage fs;
  std::string packagePath = ros::package::getPath("rs_learning") + '/';
  std::string savePath = "data/";

  if(!boost::filesystem::exists(packagePath + savePath))
  {
    std::cout << "Train data can be not found" << std::endl;

  }

  fs.open(packagePath + savePath + matrix_name + ".yaml", cv::FileStorage::READ);
  fs[matrix_name] >> des_matrix;

  fs.open(packagePath + savePath + label_name + ".yaml", cv::FileStorage::READ);
  fs [label_name] >> des_label;

}

// To show the confusion matrix and accuracy result...........................

void RSClassifier::evaluation(std::vector<int> test_label, std::vector<int> predicted_label, std::string obj_classInDouble)
{

  std::map < std::string, double > object_label;

  std::string resourcePath;
  std::string lebel_path = "data/" + obj_classInDouble + ".txt";

  resourcePath = ros::package::getPath("rs_learning") + '/';

  if(!boost::filesystem::exists(resourcePath + lebel_path))
  {
    std::cout << "objects.txt file is not found" << std::endl;

  };

  // To read the object class names from rs_resources/object_dataset/objects.txt.......

  getLabels(resourcePath + lebel_path, object_label);

  //Declare the confusion matrix which takes test data label (test_label) and predicted_label as inputs.
  // It's size is defined by the number of classes.

  std::vector <vector<int> >confusion_matrix(object_label.size(), vector<int>(object_label.size(), 0));

  for(int i = 0; i < test_label.size(); i++)
  {
    confusion_matrix[test_label[i] - 1][predicted_label[i] - 1] = confusion_matrix[test_label[i] - 1][predicted_label[i] - 1] + 1;
  }

  //To show confusion matrix .............................................

  std::cout << "confusion_matrix:" << std::endl;
  for(int i = 0; i < object_label.size(); i++)
  {
    for(int j = 0; j < object_label.size(); j++)
    {
      std::cout << confusion_matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
  //calculation of classifier accuracy........................................
  double c = 0;
  for(int i = 0; i < object_label.size(); i++)
  {
    c = c + confusion_matrix[i][i];
  }
  double Accuracy = (c / test_label.size()) * 100;
  std::cout << "classifier Accuray:" << Accuracy << std::endl;

}


//To save or load the classifier's trained data in rs_learning/data/trainedData....

std::string RSClassifier::saveOrLoadTrained(std::string trained_file_name)
{
  std::string packagePath;
  std::string save_train = "data/trainedData/";

  packagePath = ros::package::getPath("rs_learning") + '/';

  if(!boost::filesystem::exists(packagePath + save_train))
  {
    std::cout << "Trained data can not recognize" << std::endl;
  }
  std::string a = packagePath + save_train + trained_file_name + ".xml";

  return a;
}

void RSClassifier::drawCluster(cv::Mat input , cv::Rect rect, const std::string &label)
{
  cv::rectangle(input, rect, CV_RGB(0, 255, 0), 2);
  int offset = 15;
  int baseLine;
  cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1.5, 2.0, &baseLine);
  cv::putText(input, label, cv::Point(rect.x + (rect.width - textSize.width) / 2, rect.y - offset - textSize.height), cv::FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 0, 0), 2.0);

}

/*
void  RSClassifier::processPclFeature(std::string memory_name, std::vector<rs::Cluster> clusters, RSClassifier *po, cv::Mat &color)
{


  outInfo("Number of cluster:" << clusters.size() << std::endl);

  for(size_t i = 0; i < clusters.size(); ++i)
  {

    rs::Cluster &cluster = clusters[i];
    std::vector<rs::PclFeature> features;
    cluster.annotations.filter(features);

    for(size_t j = 0; j < features.size(); ++j)
    {
      rs::PclFeature &feats = features[j];

      outInfo("type of feature:" << feats.feat_type() << std::endl);
      std::vector<float> featDescriptor = feats.feature();
      outInfo("Size after conversion:" << featDescriptor.size());
      cv::Mat test_mat(1, featDescriptor.size(), CV_32F);
      for(size_t k = 0; k < featDescriptor.size(); ++k)
      {

        test_mat.at<float>(0, k) = featDescriptor[k];

      }
      outInfo("number of elements in :" << i << std::endl);

      double classLabel;
      po->classifyOnVideo(memory_name, test_mat, classLabel);

      int classLabelInInt = classLabel;
      std::string sr = "cluster_" + std::to_string(i) + '_' + std::to_string(classLabelInInt);

      //draw result on image
      rs::ImageROI image_roi = cluster.rois.get();
      cv::Rect rect;
      rs::conversion::from(image_roi.roi_hires.get(), rect);

      po->drawCluster(color, rect, sr);

      outInfo("calculation is done" << std::endl);
    }

  }


}

*/

void  RSClassifier::processVFHFeature(std::string memory_name,std::string set_mode, std::string dataset_use,std::string feature_use,
                                      std::vector<rs::Cluster> clusters, RSClassifier *obj_VFH, cv::Mat &color,std::vector<std::string> models_label, uima::CAS &tcas)
{


  outInfo("Number of cluster:" << clusters.size() << std::endl);

  for(size_t i = 0; i < clusters.size(); ++i)
  {

    rs::Cluster &cluster = clusters[i];
    std::vector<rs::PclFeature> features;
    cluster.annotations.filter(features);

    for(size_t j = 0; j < features.size(); ++j)
    {
      rs::PclFeature &feats = features[j];

      outInfo("type of feature:" << feats.feat_type() << std::endl);
      std::vector<float> featDescriptor = feats.feature();
      outInfo("Size after conversion:" << featDescriptor.size());
      cv::Mat test_mat(1, featDescriptor.size(), CV_32F);
      for(size_t k = 0; k < featDescriptor.size(); ++k)
      {

        test_mat.at<float>(0, k) = featDescriptor[k];

      }
      outInfo("number of elements in :" << i << std::endl);

      double classLabel;
      obj_VFH->classifyOnLiveData(memory_name, test_mat, classLabel);

      int classLabelInInt = classLabel;
     // std::string sr = "cluster_" + std::to_string(i) + '_' + std::to_string(classLabelInInt);
      std::string classLabelInString = models_label[classLabelInInt-1];

 //To annotate the clusters..................

       RsAnnotation (tcas,classLabelInString,feature_use, dataset_use, cluster,set_mode);

      //set roi on image
      rs::ImageROI image_roi = cluster.rois.get();
      cv::Rect rect;
      rs::conversion::from(image_roi.roi_hires.get(), rect);


      //Draw result on image...........
      obj_VFH->drawCluster(color, rect, classLabelInString);

      outInfo("calculation is done" << std::endl);
    }

  }


}


//the function process and classify RGB images, which run from a .bag file.
void  RSClassifier::processCaffeFeature(std::string memory_name, std::string set_mode, std::string dataset_use,std::string feature_use,
                                      std::vector<rs::Cluster> clusters,
                                      RSClassifier *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas)
{

  // clusters comming from RS pipeline............................
  outInfo("Number of cluster:" << clusters.size() << std::endl);

  for(size_t i = 0; i < clusters.size(); ++i)
  {

      rs::Cluster &cluster = clusters[i];
      std::vector<rs::Features> features;
      cluster.annotations.filter(features);

    for(size_t j = 0; j < features.size(); ++j)
    {
      rs::Features &feats = features[j];

      outInfo("type of feature:" << feats.descriptorType() << std::endl);
      outInfo("size of feature:" << feats.descriptors << std::endl);
      outInfo("size of source:" << feats.source() << std::endl);

      // variable to store caffe feature..........
      cv::Mat featDescriptor;

      double classLabel;

       //TODO: check the source of the feature...actually check the type as well such that it matches the model
      if(feats.source()=="Caffe")
      {
        rs::conversion::from(feats.descriptors(), featDescriptor);
        outInfo("Size after conversion:" << featDescriptor.size());


       //The function generate the prediction result................
         obj_caffe->classifyOnLiveData(memory_name, featDescriptor, classLabel);

         //class label in integer, which is used as index of vector model_label.
          int classLabelInInt = classLabel;
          std::string classLabelInString = models_label[classLabelInInt-1];

     //To annotate the clusters..................

           RsAnnotation (tcas,classLabelInString,feature_use, dataset_use, cluster,set_mode);


          //set roi on image
          rs::ImageROI image_roi = cluster.rois.get();
          cv::Rect rect;
          rs::conversion::from(image_roi.roi_hires.get(), rect);

      //Draw result on image...........................
         obj_caffe->drawCluster(color, rect, classLabelInString);

      }
      outInfo("calculation is done" << std::endl);
    }

  }


}


RSClassifier::~ RSClassifier()
{
}
