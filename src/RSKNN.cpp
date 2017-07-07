#include<iostream>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <map>
#include <yaml-cpp/yaml.h>
#include<ros/package.h>
#include<boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <rs_learning/RSClassifier.h>
#include <rs_learning/RSKNN.h>

#include <uima/api.hpp>
#include <rs/scene_cas.h>
//RS
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>

using namespace cv;



//..............................k-Nearest Neighbor Classifier.........................................

RSKNN::RSKNN()
{

}

//To train the k-Nearest Neighbor..................................
void RSKNN:: trainModel(std::string train_matrix_name, std::string train_label_name,std::string train_label_n)
{


}


void RSKNN:: classify(std::string trained_file_name_saved, std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{

}

void RSKNN::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det)
{


}



void RSKNN:: classifyKNN(std::string train_matrix_name,std::string train_label_name,
               std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{

    //To load the train data................
    cv::Mat train_matrix;
    cv::Mat train_label;
    readDescriptorAndLabel(train_matrix_name, train_label_name, train_matrix, train_label);
    std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
    std::cout << "size of train label:" << train_label.size() << std::endl;


  // To load the test data.............................
  cv::Mat test_matrix;
  cv::Mat test_label;
  readDescriptorAndLabel(test_matrix_name, test_label_name, test_matrix, test_label);
  std::cout << "size of test matrix :" << test_matrix.size() << std::endl;
  std::cout << "size of test label" << test_label.size() << std::endl;

  //value of k-neighbors........................

  int k=3;

  CvKNearest* knncalld = new CvKNearest;

  //Train the classifier...................................
        knncalld->train(train_matrix, train_label, cv::Mat(), false, k,false);

  //To load the trained data................................
 // knncls->load((saveOrLoadTrained(trained_file_name_saved)).c_str());

  //To get the value of k.............
   int k_max = knncalld->get_max_k();
  // cv::Mat neighborResponses, bestResponse, distances;

  //convert test label matrix into a vector.......................
  std::vector<double> con_test_label;
  test_label.col(0).copyTo(con_test_label);

  // Container to hold the integer value of labels............................
  std::vector<int> actual_label;
  std::vector<int> predicted_label;

  for(int i = 0; i < test_label.rows; i++)
  {
   // double res = knncls->find_nearest(test_matrix.row(i), k,bestResponse,neighborResponses,distances);
       double res = knncalld->find_nearest(test_matrix.row(i),k_max);
    int prediction = res;
    predicted_label.push_back(prediction);
    double lab = con_test_label[i];
    int actual_convert = lab;
    actual_label.push_back(actual_convert);

    //  std::cout<<"actuall class is:"<<actual_convert<<"::"<<"prediction is :"<< prediction<<std::endl;
  }
  std::cout << "K-Nearest Neighbor Result :" << std::endl;
  evaluation(actual_label, predicted_label, obj_classInDouble);

}



void RSKNN::classifyOnLiveDataKNN(std::string train_matrix_name, std::string train_label_name, cv::Mat test_mat, double &det)
{


    //To load the train data................
    cv::Mat train_matrix;
    cv::Mat train_label;
    readDescriptorAndLabel(train_matrix_name, train_label_name, train_matrix, train_label);
    std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
    std::cout << "size of train label:" << train_label.size() << std::endl;

  // To load the test data and it's label.............................

  std::cout << "size of test matrix :" << test_mat.size() << std::endl;

  int k=3;

  CvKNearest* knncalldc = new CvKNearest;

  //Train the classifier...................................
        knncalldc->train(train_matrix, train_label, cv::Mat(), false, k,false);

  //To load the trained data................................
 // knncls->load((saveOrLoadTrained(trained_file_name_saved)).c_str());

  //To get the value of k.............
   int k_max = knncalldc->get_max_k();

  //  cv::Mat neighborResponses, bestResponse, distances;

 // double res = knnclsa->find_nearest(test_mat, k, bestResponse, neighborResponses, distances);

    double res = knncalldc->find_nearest(test_mat,k_max);
  std::cout << "prediction class is :" << res << std::endl;

  det = res;

}


void  RSKNN::processVFHFeatureKNN(std::string train_matrix_name,std::string train_label_name,std::string set_mode, std::string dataset_use,std::string feature_use,
     std::vector<rs::Cluster> clusters, RSKNN *obj_VFH, cv::Mat &color,std::vector<std::string> models_label, uima::CAS &tcas)
{
    //To load the train data................

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
     // obj_VFH->classifyOnLiveData(memory_name, test_mat, classLabel);
         obj_VFH->classifyOnLiveDataKNN(train_matrix_name, train_label_name, test_mat, classLabel);
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
void  RSKNN::processCaffeFeatureKNN(std::string train_matrix_name,std::string train_label_name,
       std::string set_mode, std::string dataset_use,std::string feature_use, std::vector<rs::Cluster> clusters,
        RSKNN *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas)
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
         obj_caffe->classifyOnLiveDataKNN(train_matrix_name,train_label_name, featDescriptor, classLabel);

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



void RSKNN::RsAnnotation(uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode)
{
  rs::Classification classResult = rs::create<rs::Classification>(tcas);
  classResult.classname.set(class_name);
  classResult.classifier("k-Nearest Neighbor");
  classResult.featurename(feature_name);
  classResult.model(database_name);

  if(set_mode == "CL")
  {
//To annotate the clusters..................
  cluster.annotations.append(classResult);
     }

  else if(set_mode == "GT")
  {
      rs::GroundTruth setGT = rs::create<rs::GroundTruth>(tcas);
       setGT.classificationGT.set(classResult);
       cluster.annotations.append(setGT);
  }


  else {
       outInfo("You should set the parameter (set_mode) as CL or GT");
     }


}



RSKNN::~ RSKNN()
{

}
