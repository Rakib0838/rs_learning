

#include<iostream>
#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>
#include<ros/package.h>
#include <rs/recognition/CaffeProxy.h>
#include <dirent.h>
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include<algorithm>
#include <iterator>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

namespace po = boost::program_options;

//To read the object label from rs_resource/object/datasets folder
void readClassLabel( std::string obj_file_path, std::vector <std::pair < string, double> > &objectToLabel,
std::vector<double> &classLabelInDouble)

{
    cv::FileStorage fs;
    fs.open(obj_file_path, cv::FileStorage::READ);
    std::vector<std::string> classes;

    fs["classes"] >> classes;

    if(classes.empty())
    {
      std::cout << "Object file has no classes defined" << std::endl;

    }
    else
    {
      for(auto c : classes)

      {   double clslabel = clslabel+1;

          classLabelInDouble.push_back(clslabel);

           std::vector<std::string> subclasses;
           fs[c] >> subclasses;

        if(!subclasses.empty())
          for(auto sc : subclasses)
          {
              objectToLabel.push_back(std::pair< std::string,float >(c+'/'+sc , clslabel ));
          }
        else
        {

             objectToLabel.push_back(std::pair< std::string,float >(c , clslabel ));
        }

      }
    }

    fs.release();

         std::cout<<"objects and it's class labels:"<<std::endl;
      for(int i=0; i<objectToLabel.size(); i++){
          std::cout<< objectToLabel[i].first <<"::"<<objectToLabel[i].second<< std::endl;
      }

      std::cout<<"classLabelInDouble:"<<std::endl;

      for(int i=0; i< classLabelInDouble.size(); i++){
          std::cout<<classLabelInDouble[i]<< std::endl;
      }

}



//To read all the objects from rs_resources/objects_dataset folder.....................
void getFiles(const std::string &path, std::vector <std::pair < string, double> > object_label,
          std::map<double, std::vector<std::string> > &modelFiles, std::string file_extension)
{
  DIR *classdp;
  struct dirent *classdirp;
  size_t pos;

  for(auto const & p : object_label)
  {
        std::string pathToObj = path + p.first;
        classdp = opendir(pathToObj.c_str());
    
    if(classdp == NULL)
    {
      std::cout<< "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      std::cout<< "FOLDER DOES NOT EXIST: " << pathToObj << std::endl;
      std::cout << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      continue;
    }

    while((classdirp = readdir(classdp)) != NULL)
    {
      if(classdirp->d_type != DT_REG)
      {
        continue;
      }
      
        std::string filename = classdirp->d_name;

      pos = filename.rfind(file_extension.c_str());
      if(pos != std::string::npos)
      {
        modelFiles[p.second].push_back(pathToObj + '/' + filename);

      }

    }
  }

 std::map<double, std::vector<std::string> >::iterator it;

 for(it = modelFiles.begin(); it != modelFiles.end(); ++it)
  {
    std::sort(it->second.begin(), it->second.end());
  }

}

// To extract the VFH feature.........................................
// Taken from rs_addons...............................................
void extractVFHDescriptors(const std::map<double, std::vector<std::string> > &modelFiles,
                           std::vector<std::pair<double, std::vector<float> > > &vfh_features)
{
 
  pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfhEstimation;
  for(std::map<double, std::vector<std::string> >::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it)
  {
    std::cerr << it->first << std::endl;
    for(int i = 0; i < it->second.size(); ++i)
    {
      std::cerr << it->second[i] << std::endl;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::io::loadPCDFile(it->second[i], *cloud);

      pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
      ne.setInputCloud(cloud);

      pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA> ());
      ne.setSearchMethod(tree);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
      ne.setRadiusSearch(0.03);
      ne.compute(*cloud_normals);

      vfhEstimation.setInputCloud(cloud);
      vfhEstimation.setInputNormals(cloud_normals);
      vfhEstimation.setNormalizeBins(true);
      vfhEstimation.setNormalizeDistance(true);
      vfhEstimation.setSearchMethod(tree);

      pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(new pcl::PointCloud<pcl::VFHSignature308> ());
      vfhEstimation.compute(*vfhs);
      std::vector<float> vfhsVec;
      vfhsVec.resize(308);
      for(size_t j = 0; j < 308; ++j)
      {
        vfhsVec[j] = vfhs->points[0].histogram[j];
      }
      vfh_features.push_back(std::pair<double, std::vector<float> >(it->first, vfhsVec));
    }
  }

  std::cerr << "vfh_features size: " << vfh_features.size() << std::endl;
}

// To extract the CNN feature.........................................
// Taken from rs_addons...............................................
void extractCNNFeature(const  std::map<double, std::vector<std::string> > &modelFiles,
                       std::string resourcesPackagePath,
                       std::vector<std::pair<double, std::vector<float> > > &cnn_features)
{

std::string CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
std::string  CAFFE_TRAINED_FILE= "/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
std::string CAFFE_MEAN_FILE= "/caffe/data/imagenet_mean.binaryproto";
std::string CAFFE_LABLE_FILE = "/caffe/data/synset_words.txt";



 CaffeProxy caffeProxyObj(resourcesPackagePath + CAFFE_MODEL_FILE,
                           resourcesPackagePath + CAFFE_TRAINED_FILE,
                           resourcesPackagePath + CAFFE_MEAN_FILE,
                           resourcesPackagePath + CAFFE_LABLE_FILE);

  for(std::map<double, std::vector<std::string>>::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it)
  {
    std::cerr << it->first << std::endl;
    for(int i = 0; i < it->second.size(); ++i)
    {
      std::cerr << it->second[i] << std::endl;
      cv::Mat rgb = cv::imread(it->second[i]);
      std::vector<float> feature = caffeProxyObj.extractFeature(rgb);

      cv::Mat desc(1, feature.size(), CV_32F, &feature[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart, (float *)desc.dataend);
      cnn_features.push_back(std::pair<double, std::vector<float>>(it->first, descNormed));
    }
  }
  std::cerr << "cnn_features size: " << cnn_features.size() << std::endl;
}


// To split the instance dataset into train and and test dataset..............................
void splitDataset(std::vector<std::pair<double, std::vector<float> > > features,
                  std::vector<std::pair<double, std::vector<float> > > &output_train,
                  std::vector<std::pair<double, std::vector<float> > > &output_test)
{

     // The following loop split every fourth desccriptor and store it to vector output_test
    for(int i=0; i<features.size()/4;i++)
    {
        for(int j=0; j<3; j++ ){
            output_train.push_back(features[j+4*i]);
        }
           output_test.push_back(features[3+4*i]);
    }

}


// To split the object's dataset into train and and test dataset..............................
void SplitObjectLabel(std::vector <std::pair < string, double> > objToLab,
                  std::vector <std::pair < string, double> > &obj_train,
                  std::vector <std::pair < string, double> > &obj_test)
{

     // The following loop split every fourth desccriptor and store it to vector output_test
    for(int i=0; i<objToLab.size()/3;i++)
    {
        for(int j=0; j<2; j++ ){
            obj_train.push_back(std::pair< std::string,float > (objToLab[j+3*i].first ,objToLab[j+3*i].second));
        }
           obj_test.push_back(std::pair< std::string,float > (objToLab[2+3*i].first ,objToLab[2+3*i].second));
    }


    std::cout<<"objectTolabel_train:"<<std::endl;

       for(int i=0; i< obj_train.size(); i++){
        std::cout<< obj_train[i].first <<"::"<< obj_train[i].second<< std::endl;
       }


    std::cout<<"objectTolabel_test:"<<std::endl;

    for(int i=0; i<  obj_test.size(); i++){
        std::cout<<  obj_test[i].first <<"::"<<  obj_test[i].second<< std::endl;
    }



}

// To save the train and test data in cv::Mat format in folder /rs_learning/data
void saveDatasets (std::vector<std::pair<double, std::vector<float> > > train_dataset,
                   std::vector<std::pair<double, std::vector<float> > > test_dataset ,std::string split_name, std::string descriptor_name  )
{
    cv::Mat descriptors_train (train_dataset.size(), train_dataset[0].second.size(), CV_32F);
    cv::Mat label_train (train_dataset.size(), 1, CV_32F);

    cv::Mat descriptors_test (test_dataset.size(), test_dataset[0].second.size(), CV_32F);
    cv::Mat label_test (test_dataset.size(), 1, CV_32F);

    for(size_t i = 0; i <  train_dataset.size(); ++i)
     {
          label_train.at<float>(i,0)= train_dataset[i].first;

        for(size_t j = 0; j <  train_dataset[i].second.size(); ++j)
          {
            descriptors_train.at<float>(i, j) =  train_dataset[i].second[j];
          }
      }


    for(size_t i = 0; i <  test_dataset.size(); ++i)
     {
          label_test.at<float>(i,0)= test_dataset[i].first;

        for(size_t j = 0; j <  test_dataset[i].second.size(); ++j)
          {
            descriptors_test.at<float>(i, j) =  test_dataset[i].second[j];
          }
      }


    //To save file in disk...........................................................

     cv::FileStorage fs;
      std::string packagePath = ros::package::getPath("rs_learning");
       std::string savePath = packagePath + "/data/";

           // To check the resource path................................................
       if(!boost::filesystem::exists(savePath))
          {
           std::cout<<"folder called data is not found to save the training data"<<std::endl;
          }

       // To save the train data.................................................

       fs.open(savePath +"Mat_train_"+ split_name +'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
       fs <<"Mat_train_"+split_name+'_'+ descriptor_name << descriptors_train;
          fs.release();
    fs.open(savePath + "label_train_"+split_name+'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
       fs <<"label_train_"+ split_name+'_'+ descriptor_name << label_train;
      fs.release();

      // To save the test data.....................................................

      fs.open(savePath +"Mat_test_"+split_name+'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
      fs <<"Mat_test_"+split_name+'_'+ descriptor_name << descriptors_test;
         fs.release();
   fs.open(savePath + "label_test_"+split_name+'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
      fs <<"label_test_"+split_name+'_'+ descriptor_name << label_test;
     fs.release();
}


// To save the train and test data in cv::Mat format in folder /rs_learning/data
void saveAallDescriptors (std::vector<std::pair<double, std::vector<float> > > features_vec,
                   std::string split_name, std::string descriptor_name )
{
    cv::Mat descriptors (features_vec.size(), features_vec[0].second.size(), CV_32F);
    cv::Mat descriptors_label (features_vec.size(), 1, CV_32F);


    for(size_t i = 0; i < features_vec.size(); ++i)
     {
          descriptors_label.at<float>(i,0)= features_vec[i].first;

        for(size_t j = 0; j <  features_vec[i].second.size(); ++j)
          {
            descriptors.at<float>(i, j) =  features_vec[i].second[j];
          }
      }


    //To save file in disk...........................................................

     cv::FileStorage fs;
      std::string packagePath = ros::package::getPath("rs_learning");
       std::string savePath = packagePath + "/data/";

           // To check the resource path................................................
       if(!boost::filesystem::exists(savePath))
          {
           std::cout<<"folder called data is not found to save the training data"<<std::endl;
          }

       // To save the train data.................................................

       fs.open(savePath +"Descriptors_"+ split_name +'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
       fs <<"Descriptors_"+ split_name +'_'+ descriptor_name << descriptors;
          fs.release();
    fs.open(savePath +"Descriptors_label_"+split_name +'_'+ descriptor_name + ".yaml", cv::FileStorage::WRITE);
       fs <<"Descriptors_label_"+ split_name +'_'+ descriptor_name << descriptors_label;
      fs.release();
}


//To save class labels in type double in folder rs_learning/data ................
void saveClassLabels(std::vector<double> input_file, std::string split_name, std::string feat_name)
{
    std::string packagePath = ros::package::getPath("rs_learning");
     std::string savePath = packagePath + "/data/";

         // To check the resource path................................................
     if(!boost::filesystem::exists(savePath))
        {
         std::cout<<"folder called data is not found to save the <<< classLabel in Double data >>>"<<std::endl;
        }


    std::ofstream file((savePath+"ClassLabel_"+split_name+'_'+feat_name+".txt").c_str());

    for(auto p: input_file)
   {
       file<<p<<endl;
   }
}

int main(int argc, char **argv)

{
    po::options_description desc("Allowed options");
    std::string objects_name,storage, feat, split;
    desc.add_options()
    ("help,h", "Print help messages")
    ("file,f", po::value<std::string>(& objects_name)->default_value("ob_wu"),
     "enter the object file name")
    ("storage,s", po::value<std::string>(& storage)->default_value("png_wu"),
    "enter storage folder name")
    ("split,o", po::value<std::string>(&split)->default_value("OBJ"),
    "choose way to split: [OBJ|INS|ALL]")
    ("feature,r", po::value<std::string>(&feat)->default_value("CNN"),
     "choose feature to extract: [CNN|VFH]");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
      std::cout << desc << "\n";
      return 1;
    }


    // Define path to get the datasets.......................................................
    std::string resourcePath = ros::package::getPath("rs_resources");
     std::string object_file_path = objects_name;
     std::string model_files_path = storage;

    if(!boost::filesystem::exists(boost::filesystem::path(objects_name)) ||
       !boost::filesystem::exists(boost::filesystem::path(storage)) )
    {
      object_file_path = resourcePath + "/objects_dataset/" + objects_name + ".yaml";
      model_files_path =  resourcePath + "/objects_dataset/" + storage+'/';
    }


      std::cout << "Path to object label : " << object_file_path << std::endl;
     std::cout << "Path of object storage : " << model_files_path  << std::endl;


    std::vector <std::pair < string, double> > objectToLabel;
    std::vector<double> classLabelInDouble;

    readClassLabel(object_file_path, objectToLabel, classLabelInDouble);

    //................................................................................................

   //.................................................................................................


        // split (objectToLabel) to create train an test dataset...........
       std::vector <std::pair < string, double> > objectToLabel_train;
       std::vector <std::pair < string, double> > objectToLabel_test;

       // need to store .pcd  or .png file from storage
       std::map< double, std::vector<std::string> > model_files_all;
       std::map< double, std::vector<std::string> > model_files_train;
       std::map< double, std::vector<std::string> > model_files_test;

       //Extract the feat descriptors
       std::vector<std::pair<double, std::vector<float> > > descriptors_all;
       std::vector<std::pair<double, std::vector<float> > > descriptors_train;
       std::vector<std::pair<double, std::vector<float> > > descriptors_test;

       //To store splitted train and test descriptors
       std::vector<std::pair<double, std::vector<float> > > descriptors_all_train;
       std::vector<std::pair<double, std::vector<float> > > descriptors_all_test;



            if(split == "INS" && feat == "VFH")
                   {
                      std::cout<<"Starts calculation with instance(INS) and VFH :"<<std::endl;

                      // To read all .cpd files from the storage folder...........
                      getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                      // To calculate VFH descriptors..................................
                      extractVFHDescriptors(model_files_all, descriptors_all);

                      // To split all the calculated VFH descriptors (descriptors_all) into train and test data for
                      // the classifier. Here evey fourth element of vector (descriptors_all) is considered as test data
                      // and rest are train data
                      splitDataset(descriptors_all, descriptors_all_train, descriptors_all_test);

                      // To save the train and test data in folder /rs_learning/data
                      saveDatasets (descriptors_all_train, descriptors_all_test, split, feat);

                      // To save the class labels in type double in folder /rs_learning
                      saveClassLabels(classLabelInDouble, split, feat);
                   }

            else if(split == "INS" && feat == "CNN")
                   {
                      std::cout<<"Starts calculation with instance(INS) and CNN :"<<std::endl;

                      // To read all .png files from the storage folder...........
                     getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

                     // To calculate VFH descriptors..................................
                     extractCNNFeature(model_files_all, resourcePath, descriptors_all);

                     // To split all the calculated VFH descriptors (descriptors_all) into train and test data for
                     // the classifier. Here evey fourth element of vector (descriptors_all) is considered as test data
                     // and rest are train data
                    splitDataset(descriptors_all, descriptors_all_train, descriptors_all_test);

                    // To save the train and test data in folder /rs_learning/data
                      saveDatasets (descriptors_all_train, descriptors_all_test, split, feat);

                    // To save the class labels in type double in folder rs_learning/data
                     saveClassLabels(classLabelInDouble, split, feat);
                   }
     
          else if(split == "OBJ" && feat == "VFH")
                {
                    std::cout<<"Starts calculation with object(OBJ) and VFH :"<<std::endl;

                   // Split the object labels to create the train and test data, where every third object of the sub-class
                    //consider as the test data. which is created to  work with washington university's datasets
                   SplitObjectLabel(objectToLabel, objectToLabel_train, objectToLabel_test);

                    // To read .cpd files from the storage folder...........
                   getFiles(model_files_path, objectToLabel_train, model_files_train, ".pcd");

                   // To read .cpd files from the storage folder...........
                   getFiles(model_files_path, objectToLabel_test, model_files_test, ".pcd");

                    // To calculate VFH descriptors..................................
                   extractVFHDescriptors(model_files_train, descriptors_train);

                    // To calculate VFH descriptors..................................
                   extractVFHDescriptors(model_files_test, descriptors_test);

                    // To save the train and test data in folder /rs_learning/data
                   saveDatasets (descriptors_train, descriptors_test, split, feat);

                   // To save the class labels in type double in folder /rs_learning/data
                    saveClassLabels(classLabelInDouble, split, feat);
                }

          else if(split == "OBJ" && feat == "CNN")
                {
                    std::cout<<"Starts calculation with object(OBJ) and CNN :"<<std::endl;

                    // Split the object labels to create the train and test data, where every third object of the sub-class
                     //consider as the test data. which is created to  work with washington university's datasets
                    SplitObjectLabel(objectToLabel, objectToLabel_train, objectToLabel_test);

                    // To read .png files from the storage folder...........
                    getFiles(model_files_path, objectToLabel_train, model_files_train, "_crop.png");

                    // To read .png files from the storage folder...........
                    getFiles(model_files_path, objectToLabel_test, model_files_test, "_crop.png");

                    // To calculate CNN features..................................
                    extractCNNFeature(model_files_train, resourcePath, descriptors_train);

                    // To calculate CNN features..................................
                    extractCNNFeature(model_files_test, resourcePath, descriptors_test);

                     // To calculate CNN features..................................
                    saveDatasets (descriptors_train, descriptors_test, split, feat);

                    // To save the class labels in type double in folder /rs_learning/data
                    saveClassLabels(classLabelInDouble, split, feat);
                }


          else if(split == "ALL" && feat == "VFH")
                   {
                      std::cout<<"Starts calculation with all (ALL) and VFH :"<<std::endl;

                      // To read all .cpd files from the storage folder...........
                      getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                      // To calculate all VFH descriptors..................................
                        extractVFHDescriptors(model_files_all, descriptors_all);

                      // To save all descriptors in folder /rs_learning/data
                         saveAallDescriptors (descriptors_all, split, feat );

                      // To save the class labels in type double in folder /rs_learning
                      saveClassLabels(classLabelInDouble, split, feat);
                   }


          else if(split == "ALL" && feat == "CNN")
             {
                std::cout<<"Starts calculation with all (ALL) and CNN :"<<std::endl;

                // To read all .png files from the storage folder...........
               getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

               // To calculate CNN descriptors..................................
               extractCNNFeature(model_files_all, resourcePath, descriptors_all);

              // To save all descriptors in folder /rs_learning/data
               saveAallDescriptors (descriptors_all, split, feat );

              // To save the class labels in type double in folder rs_learning/data
               saveClassLabels(classLabelInDouble, split, feat);
             }

     std::cout<<"Descriptors calculation is done"<<std::endl;


    return 0;
}  
