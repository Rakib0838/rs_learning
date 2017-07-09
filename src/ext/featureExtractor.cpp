

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
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>




using namespace cv;
using namespace std;

namespace po = boost::program_options;


//To read the object label from rs_resource/object_datasets folder
void readClassLabelIAI( std::string obj_file_path, std::vector <std::pair < string, double> > &objectToLabel,
                        std::vector <std::pair < string, double> > &objectToClassLabelMap)

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

           std::vector<std::string> subclasses;
            fs[c] >> subclasses;

            //To set the map between string and double classlabel
            objectToClassLabelMap.push_back(std::pair< std::string,float >(c , clslabel ));

        if(!subclasses.empty())
          for(auto sc : subclasses)
          {                       

                    objectToLabel.push_back(std::pair< std::string,float >(sc , clslabel ));
          }
        else
        {

             objectToLabel.push_back(std::pair< std::string,float >(c , clslabel ));
        }

      }
    }

    fs.release();


    if(!objectToClassLabelMap.empty()) {
   std::cout<<"objectToClassLabelMap:"<<std::endl;

   for(int i=0; i<objectToClassLabelMap.size(); i++){
       std::cout<< objectToClassLabelMap[i].first <<"::"<<objectToClassLabelMap[i].second<< std::endl;
         }

    }


    if(!objectToLabel.empty()) {
     std::cout<<"objectToLabel:"<<std::endl;
  for(int i=0; i<objectToLabel.size(); i++){
      std::cout<< objectToLabel[i].first <<"::"<<objectToLabel[i].second<< std::endl;
          }

    }



}


//To read the object label from rs_resource/object_datasets folder
//To read the .yml for Washington Uni........................................
void readClassLabelWU( std::string obj_file_path, std::vector <std::pair < string, double> > &objectToLabelTrain,
           std::vector <std::pair < string, double> > &objectToLabelTest, std::vector <std::pair < string, double> > &objectToClassLabelMap)

{
    std::vector<std::string> subclasses;
     std::vector<std::string> subsubclasses;


     std::vector <std::pair < string, double> > objToLabTestSubcls;
     std::vector <std::pair < string, double> > objToLabTrainSubcls;

     std::vector <std::pair < string, double> > objToLabTestSubSubcls;
     std::vector <std::pair < string, double> > objToLabTrainSubSubcls;

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

         //  std::vector<std::string> subclasses;
            fs[c] >> subclasses;

            //To set the map between string and double classlabel
            objectToClassLabelMap.push_back(std::pair< std::string,float >(c , clslabel ));

          if(!subclasses.empty())
               {


              for(int j=0; j<subclasses.size(); j++)

                     {

                        if(j==0)
                          {
                               objToLabTestSubcls.push_back(std::pair< std::string,float >(c+'/'+subclasses[j] , clslabel ));
                           }

                               else
                                  {
                                    objToLabTrainSubcls.push_back(std::pair< std::string,float >(c+'/'+subclasses[j] , clslabel ));
                                   }
                     }


              for(auto sc : subclasses)
                 {

                     fs[sc] >> subsubclasses;

                      if(!subsubclasses.empty())
                       {
                           for(int k=0; k<subsubclasses.size(); k++)
                               {

                                   if(k==0)
                                    {
                                       objToLabTestSubSubcls.push_back(std::pair< std::string,float >(sc+'/'+subsubclasses[k] , clslabel ));
                                     }

                                        else
                                        {
                                          objToLabTrainSubSubcls.push_back(std::pair< std::string,float >(sc+'/'+subsubclasses[k] , clslabel ));
                                          }
                                }


                        }


                 }

       }

      }
    }


    fs.release();

     if(!objectToClassLabelMap.empty()) {
    std::cout<<"objectToClassLabelMap:"<<std::endl;

    for(int i=0; i<objectToClassLabelMap.size(); i++){
        std::cout<< objectToClassLabelMap[i].first <<"::"<<objectToClassLabelMap[i].second<< std::endl;
          }

     }






     if(!subsubclasses.empty() && !subclasses.empty())
     {

           for(int i=0; i<objToLabTestSubSubcls.size(); i++)
           {

             objectToLabelTest.push_back(std::pair< std::string,float >(objToLabTestSubSubcls[i].first , objToLabTestSubSubcls[i].second));

              }

           for(int i=0; i<objToLabTrainSubSubcls.size(); i++)
           {

             objectToLabelTrain.push_back(std::pair< std::string,float >(objToLabTrainSubSubcls[i].first , objToLabTrainSubSubcls[i].second));

              }

     }

     else {

         for(int i=0; i<objToLabTestSubcls.size(); i++)
         {

           objectToLabelTest.push_back(std::pair< std::string,float >(objToLabTestSubcls[i].first , objToLabTestSubcls[i].second));

            }

         for(int i=0; i<objToLabTrainSubcls.size(); i++)
         {

           objectToLabelTrain.push_back(std::pair< std::string,float >(objToLabTrainSubcls[i].first , objToLabTrainSubcls[i].second));



           }
     }



     if(!objectToLabelTest.empty()) {
         std::cout<<"objectToLabelTest:"<<std::endl;
      for(int i=0; i<objectToLabelTest.size(); i++){
          std::cout<< objectToLabelTest[i].first <<"::"<<objectToLabelTest[i].second<< std::endl;
      }
    }

     if(!objectToLabelTrain.empty()) {
      std::cout<<"objectToLabelTrain:"<<std::endl;
   for(int i=0; i<objectToLabelTrain.size(); i++){
       std::cout<< objectToLabelTrain[i].first <<"::"<<objectToLabelTrain[i].second<< std::endl;
   }

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
void extractPCLDescriptors(std::string descriptorType, const std::map<double, std::vector<std::string> > &modelFiles,
                           std::vector<std::pair<double, std::vector<float> > > &extract_features)
{
 


     std::string featDescription;
  
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

      pcl::PointCloud<pcl::VFHSignature308>::Ptr extractedDiscriptor(new pcl::PointCloud<pcl::VFHSignature308> ());



       if (descriptorType=="VFH"){


       std::cout<<"Calculation start with VFH Feature"<<std::endl;

       pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfhEstimation;
      vfhEstimation.setInputCloud(cloud);
      vfhEstimation.setInputNormals(cloud_normals);
      vfhEstimation.setNormalizeBins(true);
      vfhEstimation.setNormalizeDistance(true);
      vfhEstimation.setSearchMethod(tree);
      vfhEstimation.compute(*extractedDiscriptor);

      featDescription="VFH feature size :";

        }



       if(descriptorType=="CVFH"){

           std::cout<<"Calculation start with CVFH Feature"<<std::endl;

           pcl::CVFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> cvfhEst;
           cvfhEst.setInputCloud(cloud);
           cvfhEst.setInputNormals(cloud_normals);
           cvfhEst.setSearchMethod(tree);
           cvfhEst.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
           cvfhEst.setCurvatureThreshold(1.0);
           cvfhEst.setNormalizeBins(true);
           cvfhEst.compute(*extractedDiscriptor);

           featDescription="CVFH feature size :";
       }

      std::vector<float> descriptorVec;
      descriptorVec.resize(308);
      for(size_t j = 0; j < 308; ++j)
      {
        descriptorVec[j] = extractedDiscriptor->points[0].histogram[j];
      }
      extract_features.push_back(std::pair<double, std::vector<float> >(it->first, descriptorVec));
    }
  }

  std::cerr << featDescription << extract_features.size() << std::endl;
}



// To extract the CNN feature.........................................
// Taken from rs_addons...............................................
void extractCaffeFeature(std::string featType, const  std::map<double, std::vector<std::string> > &modelFiles,
                       std::string resourcesPackagePath,
                       std::vector<std::pair<double, std::vector<float> > > &caffe_features)
{
    std::string CAFFE_MODEL_FILE;
    std::string CAFFE_TRAINED_FILE;



  std::string featDescription;

   if(featType=="VGG16")
   { CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers_deploy.prototxt";
       CAFFE_TRAINED_FILE= "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers.caffemodel";

       featDescription="VGG16 feature size :";
   }
   
   
  else if(featType=="CNN")
   { CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
      CAFFE_TRAINED_FILE= "/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

      featDescription="CNN feature size :";
   }

   else{
       std::cout<<"CAFFE_MODEL_FILE and CAFFE_TRAINED_FILE are not found"<<std::cout;
   }

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
      caffe_features.push_back(std::pair<double, std::vector<float>>(it->first, descNormed));
    }
  }
  std::cerr << featDescription << caffe_features.size() << std::endl;
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

// To split the instance dataset into train and and test dataset..............................
void descriptorsSplit( std::vector<std::pair<double, std::vector<float> > > features,
                   std::vector<std::pair<double, std::vector<float> > > &output)
{

     // The following loop split every fourth desccriptor and store it to vector output_test
    for(int i=0; i<features.size()/4;i++)
    {
      output.push_back(features[4*i]);
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
                   std::vector<std::pair<double, std::vector<float> > > test_dataset ,
                   std::string split_name, std::string descriptor_name,
                   std::string database_name, std::string xml_filename  )
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

       fs.open(savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTrain"+'_'+ xml_filename+".yaml", cv::FileStorage::WRITE);
       fs <<database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTrain"+'_'+xml_filename << descriptors_train;
          fs.release();
    fs.open(savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTrainLabel"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
       fs <<database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTrainLabel"+'_'+xml_filename<< label_train;
      fs.release();

      // To save the test data.....................................................

      fs.open(savePath +database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTest"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
      fs <<database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTest"+'_'+xml_filename<< descriptors_test;
         fs.release();
   fs.open(savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTestLabel"+'_'+xml_filename+ ".yaml", cv::FileStorage::WRITE);
      fs <<database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"MatTestLabel"+'_'+xml_filename<< label_test;
     fs.release();
}


// To save the train and test data in cv::Mat format in folder /rs_learning/data
void saveAallDescriptors (std::vector<std::pair<double, std::vector<float> > > features_vec,
                   std::string split_name, std::string descriptor_name, std::string database_name, std::string xml_filename )
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

       fs.open(savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"Des"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
       fs << database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"Des"+'_'+xml_filename << descriptors;
          fs.release();
    fs.open(savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"DesLabel"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
       fs << database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"DesLabel"+'_'+xml_filename << descriptors_label;
      fs.release();
}


void saveObjectToLabels( std::vector <std::pair < string, double> > input_file,std::string split_name,
                         std::string descriptor_name, std::string database_name, std::string xml_filename)
{
    std::string packagePath = ros::package::getPath("rs_learning");
     std::string savePath = packagePath + "/data/";

         // To check the resource path................................................
     if(!boost::filesystem::exists(savePath))
        {
         std::cout<<"folder called data is not found to save the <<< classLabel in Double data >>>"<<std::endl;
        }


    std::ofstream file((savePath + database_name +'_'+ descriptor_name +'_'+ split_name +'_'+"ClassLabel"+'_'+xml_filename+".txt").c_str());

    for(auto p: input_file)
   {
       file<<p.first<<":"<<p.second<<endl;
   }
}

int main(int argc, char **argv)

{
    po::options_description desc("Allowed options");
    std::string objects_name,storage, feat, split,database_name;
    desc.add_options()
    ("help,h", "Print help messages")
    ("file,f", po::value<std::string>(& objects_name)->default_value("obj_10_our"),
     "enter the object file name")
    ("storage,s", po::value<std::string>(& storage)->default_value("partial_views"),
    "enter storage folder name")
    ("database,d", po::value<std::string>(&database_name)->default_value("IAI"),
            "choose the database: [IAI|WU]")
    ("split,o", po::value<std::string>(&split)->default_value("INS"),
    "choose way to split. If database is IAI choose ALL or INS: [ALL|INS|ONE]")
    ("feature,r", po::value<std::string>(&feat)->default_value("VFH"),
     "choose feature to extract: [CNN|VGG16|VFH|CVFH]");

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




    //................................................................................................


   //.................................................................................................
    std::vector <std::pair < string, double> > objectToLabel;
    std::vector <std::pair < string, double> > objectToLabel_train;
    std::vector <std::pair < string, double> > objectToLabel_test;
    std::vector <std::pair < string, double> > objectToClassLabelMap;

        // To read the class label from .yaml file................
    if(database_name == "IAI")
       {

        readClassLabelIAI(object_file_path,objectToLabel, objectToClassLabelMap);
    }
    else if(database_name == "WU"){

        readClassLabelWU(object_file_path,objectToLabel_train ,objectToLabel_test,objectToClassLabelMap);

    } else {std::cout<<"Please select your 'database_name' parameter"<<std::endl;}




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

       //To store splitted train and test descriptors
       std::vector<std::pair<double, std::vector<float> > > descriptors_train_split;
       std::vector<std::pair<double, std::vector<float> > > descriptors_test_split;




       if(database_name=="IAI"){



           if(split=="ALL"){

                    if(feat=="CNN"){

                      std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                         // To read all .png files from the storage folder...........
                          getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

                        // To calculate CNN descriptors..................................
                          extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
                                  }
                    else if(feat=="VGG16"){

                        std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                           // To read all .png files from the storage folder...........
                            getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

                          // To calculate CNN descriptors..................................
                            extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
                                    }

                    else if(feat == "VFH")
                             {
                                std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                                // To read all .cpd files from the storage folder...........
                                  getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                                // To calculate all VFH descriptors..................................
                                  extractPCLDescriptors(feat, model_files_all, descriptors_all);
                             }

                    else if(feat == "CVFH" )
                             {
                              std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                                // To read all .cpd files from the storage folder...........
                                  getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                                // To calculate all VFH descriptors..................................
                                  extractPCLDescriptors(feat, model_files_all, descriptors_all);
                             }

                              else  {
                                     std::cout<<"Please select feature (CNN , VGG16, VFH, CVFH)"<<std::endl;
                                 }

                    // To save all descriptors in folder /rs_learning/data
                      saveAallDescriptors (descriptors_all, split, feat ,database_name,objects_name);


           }

            else if(split=="INS"){

                 if(feat == "CNN")
                      {
                         std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);

                      }

               else if(feat == "VGG16")
                      {
                      std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(model_files_path, objectToLabel, model_files_all, "_crop.png");

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
                      }

               else if(feat == "VFH")
                        {
                      std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                           // To read all .cpd files from the storage folder...........
                           getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                           // To calculate VFH descriptors..................................
                           extractPCLDescriptors(feat,model_files_all, descriptors_all);
                        }

                 else if(feat == "CVFH")
                          {
                            std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                             // To read all .cpd files from the storage folder...........
                             getFiles(model_files_path, objectToLabel, model_files_all, ".pcd");

                             // To calculate VFH descriptors..................................
                             extractPCLDescriptors(feat,model_files_all, descriptors_all);
                        }


                 else  {
                        std::cout<<"Please select feature (CNN , VGG16, VFH, CVFH)"<<std::endl;
                    }

                 // To split all the calculated VFH descriptors (descriptors_all) into train and test data for
                 // the classifier. Here evey fourth element of vector (descriptors_all) is considered as test data
                 // and rest are train data
                 splitDataset(descriptors_all, descriptors_all_train, descriptors_all_test);

                 // To save the train and test data in folder /rs_learning/data
                 saveDatasets (descriptors_all_train, descriptors_all_test, split, feat,database_name,objects_name);


           }

           else {
               std::cout<<"Please select split (ALL or INS for IAI or ONE for WU database)"<<std::endl;
           }



       }
       else if(database_name=="WU"){

            //  readClassLabelWU(object_file_path,objectToLabel_train ,objectToLabel_test,objectToClassLabelMap);
           // Split the object labels to create the train and test data, where every third object of the sub-class
            //consider as the test data. which is created to  work with washington university's datasets

           if(split=="ONE"){


               if( feat == "CNN")
                   {
                       std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                       // To read .png files from the storage folder...........
                       getFiles(model_files_path, objectToLabel_train, model_files_train, "_crop.png");


                       // To read .png files from the storage folder...........
                       getFiles(model_files_path, objectToLabel_test, model_files_test, "_crop.png");

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat ,model_files_train, resourcePath, descriptors_train);

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat, model_files_test, resourcePath, descriptors_test);

                   }

              else if( feat == "VGG16")
                   {
                       std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                      // To read .png files from the storage folder...........
                       getFiles(model_files_path, objectToLabel_train, model_files_train, "_crop.png");

                       // To read .png files from the storage folder...........
                       getFiles(model_files_path, objectToLabel_test, model_files_test, "_crop.png");


                       // To calculate CNN features..................................
                       extractCaffeFeature(feat ,model_files_train, resourcePath, descriptors_train);

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat, model_files_test, resourcePath, descriptors_test);

                   }

              else if(feat == "VFH" )
                     {
                        std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                         // To read .png files from the storage folder...........
                         getFiles(model_files_path, objectToLabel_train, model_files_train, ".pcd");

                         // To read .png files from the storage folder...........
                         getFiles(model_files_path, objectToLabel_test, model_files_test, ".pcd");

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat,model_files_train, descriptors_train);

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat ,model_files_test, descriptors_test);

                     }

            else if(feat == "CVFH")
                     {
                        std::cout<<"Calculation starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;

                         // To read .png files from the storage folder...........
                         getFiles(model_files_path, objectToLabel_train, model_files_train, ".pcd");

                         // To read .png files from the storage folder...........
                         getFiles(model_files_path, objectToLabel_test, model_files_test, ".pcd");

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat, model_files_train, descriptors_train);

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat, model_files_test, descriptors_test);
                        }

                   else  {
                          std::cout<<"Please select feature (CNN , VGG16, VFH, CVFH)"<<std::endl;
                        }


               //to take every fourth elements...........................
               descriptorsSplit(descriptors_train,descriptors_train_split);
               descriptorsSplit(descriptors_test,descriptors_test_split);
              saveDatasets (descriptors_train_split, descriptors_test_split, split, feat,database_name,objects_name);
               std::cout<<"Caculan starts with :" <<database_name<<"::"<<feat<<"::"<<split<<std::endl;



           } else{ std::cout<<"Please select split (ALL or INS for IAI or ONE for WU database)"<<std::endl;}


       }  else{ std::cout<<"Please select databese name (IAI or WU)"<<std::endl;}


      // To save the string class labels in type double in folder rs_learning/data
         saveObjectToLabels(objectToClassLabelMap,split, feat,database_name, objects_name);


       std::cout<<"Descriptors calculation is done"<<std::endl;


    return 0;
}  
