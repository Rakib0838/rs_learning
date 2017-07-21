

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


//To read the split file for both the IAI dataset and BOTH datasets from rs_resource/object_datasets/ folder
void readClassLabel( std::string obj_file_path, std::vector <std::pair < string, double> > &objectToLabel,
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

             objectToLabel.push_back(std::pair< std::string,float >(c, clslabel ));
        }

      }
    }

    fs.release();


    if(!objectToClassLabelMap.empty()) {
   std::cout<<"objectToClassLabelMap:"<<std::endl;

   for(int i=0; i<objectToClassLabelMap.size(); i++){
       std::cout<< objectToClassLabelMap[i].first <<"::"<<objectToClassLabelMap[i].second<< std::endl;
         }

    }      std::cout<<std::endl;


    if(!objectToLabel.empty()) {
     std::cout<<"objectToLabel:"<<std::endl;
  for(int i=0; i<objectToLabel.size(); i++){
      std::cout<< objectToLabel[i].first <<"::"<<objectToLabel[i].second<< std::endl;
          }

    }      std::cout<<std::endl;



}





//To read the .yml for Washington Uni........................................
void readClassLabelWU( std::string obj_file_path, std::vector <std::pair < string, double> > &objectToLabelTrain,
           std::vector <std::pair < string, double> > &objectToLabelTest, std::vector <std::pair < string, double> > &objectToClassLabelMap)

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
      for(int i=0; i<classes.size();i++)

      {
          double clslabel = clslabel+1;

          objectToClassLabelMap.push_back(std::pair< std::string,float >(classes[i], clslabel ));


          std::vector<std::string> subclasses;
           fs[classes[i]] >> subclasses;

        if(!subclasses.empty())

          for(int j=0; j<subclasses.size(); j++)

          {

              if(j==0)
              {
                  objectToLabelTest.push_back(std::pair< std::string,float >(subclasses[j] , clslabel ));
              }

          else
              {
              objectToLabelTrain.push_back(std::pair< std::string,float >(subclasses[j] , clslabel ));
                }
      }

      }
    }

    fs.release();

     if(!objectToClassLabelMap.empty()) {
    std::cout<<"objectToClassLabel:"<<std::endl;

    for(int i=0; i<objectToClassLabelMap.size(); i++){
        std::cout<< objectToClassLabelMap[i].first <<"::"<<objectToClassLabelMap[i].second<< std::endl;
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



void getParentDir(std::string path_storage , std::vector<std::string>& updir_wu_folder)
{

    DIR *dir = opendir((path_storage).c_str());
       if(dir)
       {
           struct dirent *ent;
           while((ent = readdir(dir)) != NULL)
           {
                updir_wu_folder.push_back(ent->d_name);

           }

       }
       else
       {
           std::cout << "Intermediate directory folder is not found" <<std::endl;
       }


}



//To read all the objects from rs_resources/objects_dataset folder.....................
void getFiles(const std::string &resourchPath, std::string storage_fol, std::vector <std::pair < string, double> > object_label,
          std::map<double, std::vector<std::string> > &modelFiles, std::string file_extension, std::string dataset_input)
{


  std::string path_to_data =resourchPath+"/objects_dataset/";

  DIR *classdp;
  struct dirent *classdirp;
  size_t pos;

  for(auto const & p : object_label)
  {

      std::string pathToObj;


      if(dataset_input=="IAI"){

          if( !boost::filesystem::exists(boost::filesystem::path(path_to_data+storage_fol)) )
          {
             std::cout<<"*********************************************************************************************"<<std::endl;
             std::cerr<<"Input images storage folder for " <<dataset_input<<" "<<"dataset does not exist " <<std::endl;
              std::cerr<<"You have selected storage folder(" <<storage_fol<< "), which does not exist " <<std::endl;
              std::cerr << "The storage folder should be in path:" <<path_to_data<< std::endl<<std::endl;
             std::exit(0);

          }

      //  std::cout << "Path to input image storage folder for"<<" "<<dataset_input<< ":" << path_to_data+storage_fol<< std::endl<<std::endl;

          pathToObj=path_to_data+storage_fol+'/'+p.first;
      }

      else if(dataset_input== "WU") {

          if( !boost::filesystem::exists(boost::filesystem::path(path_to_data+storage_fol)) )
          {
             std::cout<<"*********************************************************************************************"<<std::endl;
             std::cerr<<"Input images storage folder for " <<dataset_input<<" "<<"dataset does not exist " <<std::endl;
             std::cerr<<"You have selected storage folder(" <<storage_fol<< "), which does not exist " <<std::endl;
              std::cerr << "The storage folder should be in path:" <<path_to_data<< std::endl<<std::endl;
              std::exit(0);
          }

           std::vector<std::string> updir_wu;

           getParentDir(path_to_data+storage_fol,updir_wu);

           for(auto const & m : updir_wu)
           {
                if( boost::filesystem::exists(boost::filesystem::path(path_to_data+storage_fol+'/'+ m+'/'+p.first)) )
                  {
                   pathToObj=path_to_data+storage_fol+'/'+m+'/'+p.first;
                 }
           }


      }     else if(dataset_input== "BOTH") {

          std::vector<std::string> split_store_folder;
          boost::split(split_store_folder, storage_fol , boost::is_any_of("/"));

          if( !boost::filesystem::exists(boost::filesystem::path(path_to_data+split_store_folder[0])) ||
                  !boost::filesystem::exists(boost::filesystem::path(path_to_data+split_store_folder[1])))
          {
             std::cout<<"*********************************************************************************************"<<std::endl;
             std::cerr<<"Input images storage folders "<<storage_fol<<" "<<"for"<<dataset_input<<" "<<"datasets do not exist" <<std::endl;
             std::cerr<<"You have to select parameter (inputStorage) as: IAIstorage/WUstorage  folders name accordingly."<<std::endl;
              std::cerr << "The storage folders should be in path:" <<path_to_data<< std::endl<<std::endl;

             std::exit(0);
          }



        if( boost::filesystem::exists(boost::filesystem::path(path_to_data+split_store_folder[0]+'/'+p.first)) )
         {
           pathToObj=path_to_data+split_store_folder[0]+'/'+p.first;

        } else  {

          std::vector<std::string> updir_wuOne;

           getParentDir(path_to_data+split_store_folder[1] ,updir_wuOne);

           for(auto const & n : updir_wuOne)
           {
                if( boost::filesystem::exists(boost::filesystem::path(path_to_data+split_store_folder[1]+'/'+ n+'/'+p.first)) )
                  {
                   pathToObj=path_to_data+split_store_folder[1]+'/'+n+'/'+p.first;

                 }

             }

           }


      } else{ std::cout<<dataset_input<< "is a rong dataset_name"<<std::endl;}




      std::cout<<"pathToObj:"<<pathToObj<<std::endl;

      classdp = opendir(pathToObj.c_str());

    if(classdp == NULL)
    {
      std::cout<< "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      std::cout<< "FOLDER DOES NOT EXIST: " << pathToObj << std::endl;
      std::cout << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      std::cerr<<"Wrong combination of storageInput( "<<storage_fol<<" ) and dataset_name ( "<<dataset_input<<" ) parameter is chosen"<<std::endl;
      std::cerr<<"Please check the help menu"<<std::endl;
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



// To save the train and test data in cv::Mat format in folder /rs_resource/extractedFeat
void saveDatasets (std::vector<std::pair<double, std::vector<float> > > train_dataset,
                   std::vector<std::pair<double, std::vector<float> > > test_dataset , std::string descriptor_name,
                   std::string dataset_name, std::string xml_filename,std::string savePathToOutput )
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

       // To save the train data.................................................

       fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+ xml_filename+".yaml", cv::FileStorage::WRITE);
       fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrain"+'_'+xml_filename << descriptors_train;
          fs.release();
    fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
       fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTrainLabel"+'_'+xml_filename<< label_train;
      fs.release();

      // To save the test data.....................................................

      fs.open(savePathToOutput +dataset_name +'_'+ descriptor_name +'_'+"MatTest"+'_'+xml_filename+".yaml", cv::FileStorage::WRITE);
      fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTest"+'_'+xml_filename<< descriptors_test;
         fs.release();
   fs.open(savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"MatTestLabel"+'_'+xml_filename+ ".yaml", cv::FileStorage::WRITE);
      fs <<dataset_name +'_'+ descriptor_name +'_'+"MatTestLabel"+'_'+xml_filename<< label_test;
     fs.release();

      std::cout<<"extracted feautres should be found in path ("<< savePathToOutput<<")"<<std::endl;
}



void saveObjectToLabels( std::vector <std::pair < string, double> > input_file,std::string descriptor_name,
                         std::string dataset_name, std::string xml_filename, std::string savePathToOutput)
{
    std::ofstream file((savePathToOutput + dataset_name +'_'+ descriptor_name +'_'+"ClassLabel"+'_'+xml_filename+".txt").c_str());

    for(auto p: input_file)
   {
       file<<p.first<<":"<<p.second<<endl;
   }

    std::cout<<" clasLabel in double should be found in path ("<< savePathToOutput<<")"<<std::endl;
}

int main(int argc, char **argv)

{
    po::options_description desc("Allowed options");
    std::string split_name,storageInput, feat,dataset_name;
    desc.add_options()
    ("help,h", "Print help messages")
    ("split file name,s", po::value<std::string>(& split_name)->default_value("object2"),
     "enter the split file name")
    ("storageInput,i", po::value<std::string>(& storageInput)->default_value("png_wu"),
    "enter input storage folder name. If want to use both storages at once provide folders name as iaiStorageFolder/wuStorageFolder")
    ("Enter name of the dataset to use,d", po::value<std::string>(&dataset_name)->default_value("WU"),
            "choose the database: [IAI|WU|BOTH]")
    ("feature,f", po::value<std::string>(&feat)->default_value("CNN"),
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

     std::string object_file_path ;

     object_file_path = resourcePath + "/objects_dataset/splits/" + split_name + ".yaml";



    if(!boost::filesystem::exists(boost::filesystem::path( object_file_path)))
    {
       std::cout<<"*********************************************************************************************"<<std::endl;
       std::cerr<<" Class label file (.yaml) is not found. Please check the path below  :"<<std::endl;
       std::cerr << "Path to class label file : " << object_file_path << std::endl<<std::endl;
       std::cerr << "The file should be in ( rs_resources/objects_datasets/splits/ ) folder "<< std::endl<<std::endl;
       return EXIT_FAILURE;
    }

    std::cout << "Path to class label file : " << object_file_path << std::endl<<std::endl;



    //To save file in disk...........................................................

      std::string savePathToOutput = resourcePath +"/objects_dataset/extractedFeat/";

           // To check the storage folder for generated files by this program ................................................
       if(!boost::filesystem::exists(savePathToOutput))
          {
           std::cerr<<"Folder called (extractedFeat) not found to save the extracted feature generated by this code."<<std::endl;
           std::cerr<<"Please create the folder in rs_resource/objects_dataset/ and name it as extractedFeat "<<std::endl;
           return EXIT_FAILURE;
          }

       std::cout << "Path to save the extracted feature : " << savePathToOutput << std::endl<< std::endl;


   //.................................................................................................
    std::vector <std::pair < string, double> > objectToLabel;
    std::vector <std::pair < string, double> > objectToLabel_train;
    std::vector <std::pair < string, double> > objectToLabel_test;
    std::vector <std::pair < string, double> > objectToClassLabelMap;


        // To read the class label from .yaml file................
    if(dataset_name == "IAI")
       {
           readClassLabel(object_file_path,objectToLabel, objectToClassLabelMap);
         }

      else if(dataset_name == "WU")
           {
              readClassLabelWU(object_file_path,objectToLabel_train ,objectToLabel_test,objectToClassLabelMap);
             }
           else if(dataset_name == "BOTH")
                {
                   readClassLabel(object_file_path,objectToLabel,objectToClassLabelMap);
                }
                   else {std::cout<<"Please select your 'dataset_name' parameter as IAI or WU or BOTH"<<std::endl;}






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



       if(dataset_name=="IAI"){

                 if(feat == "CNN")
                      {
                         std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(resourcePath,storageInput, objectToLabel, model_files_all, "_crop.png",dataset_name);

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);

                      }

               else if(feat == "VGG16")
                      {
                      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(resourcePath,storageInput, objectToLabel, model_files_all, "_crop.png",dataset_name);

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
                      }

               else if(feat == "VFH")
                        {
                      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                           // To read all .cpd files from the storage folder...........
                           getFiles(resourcePath,storageInput, objectToLabel, model_files_all, ".pcd",dataset_name);

                           // To calculate VFH descriptors..................................
                           extractPCLDescriptors(feat,model_files_all, descriptors_all);
                        }

                 else if(feat == "CVFH")
                          {
                            std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                             // To read all .cpd files from the storage folder...........
                             getFiles(resourcePath,storageInput, objectToLabel, model_files_all, ".pcd",dataset_name);

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
                 saveDatasets (descriptors_all_train, descriptors_all_test, feat,dataset_name,split_name,savePathToOutput);



       }
       else if(dataset_name=="WU"){

           // Split the object labels to create the train and test data, where every third object of the sub-class
            //consider as the test data. which is created to  work with washington university's datasets

               if( feat == "CNN")
                   {
                       std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                       // To read .png files from the storage folder...........
                       getFiles(resourcePath,storageInput, objectToLabel_train, model_files_train, "_crop.png",dataset_name);


                       // To read .png files from the storage folder...........
                       getFiles(resourcePath,storageInput, objectToLabel_test, model_files_test, "_crop.png",dataset_name);

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat ,model_files_train, resourcePath, descriptors_train);

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat, model_files_test, resourcePath, descriptors_test);

                   }

              else if( feat == "VGG16")
                   {
                       std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                      // To read .png files from the storage folder...........
                       getFiles(resourcePath,storageInput, objectToLabel_train, model_files_train, "_crop.png",dataset_name);

                       // To read .png files from the storage folder...........
                       getFiles(resourcePath,storageInput, objectToLabel_test, model_files_test, "_crop.png",dataset_name);


                       // To calculate CNN features..................................
                       extractCaffeFeature(feat ,model_files_train, resourcePath, descriptors_train);

                       // To calculate CNN features..................................
                       extractCaffeFeature(feat, model_files_test, resourcePath, descriptors_test);

                   }

              else if(feat == "VFH" )
                     {
                        std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read .png files from the storage folder...........
                         getFiles(resourcePath,storageInput, objectToLabel_train, model_files_train, ".pcd",dataset_name);

                         // To read .png files from the storage folder...........
                         getFiles(resourcePath,storageInput, objectToLabel_test, model_files_test, ".pcd",dataset_name);

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat,model_files_train, descriptors_train);

                         // To calculate CNN features..................................
                         extractPCLDescriptors(feat ,model_files_test, descriptors_test);

                     }

            else if(feat == "CVFH")
                     {
                        std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read .png files from the storage folder...........
                         getFiles(resourcePath,storageInput, objectToLabel_train, model_files_train, ".pcd",dataset_name);

                         // To read .png files from the storage folder...........
                         getFiles(resourcePath,storageInput, objectToLabel_test, model_files_test, ".pcd",dataset_name);

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
              saveDatasets (descriptors_train_split, descriptors_test_split, feat,dataset_name,split_name,savePathToOutput);


       }
         else if(dataset_name=="BOTH"){

                 if(feat == "CNN")
                      {
                         std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(resourcePath,storageInput, objectToLabel, model_files_all, "_crop.png",dataset_name);

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);

                      }

               else if(feat == "VGG16")
                      {
                      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                         // To read all .png files from the storage folder...........
                        getFiles(resourcePath,storageInput, objectToLabel, model_files_all, "_crop.png",dataset_name);

                        // To calculate VFH descriptors..................................
                        extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
                      }

               else if(feat == "VFH")
                        {
                      std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                           // To read all .cpd files from the storage folder...........
                           getFiles(resourcePath,storageInput,objectToLabel, model_files_all, ".pcd",dataset_name);

                           // To calculate VFH descriptors..................................
                           extractPCLDescriptors(feat,model_files_all, descriptors_all);
                        }

                 else if(feat == "CVFH")
                          {
                            std::cout<<"Calculation starts with :" <<dataset_name<<"::"<<feat<<std::endl;

                             // To read all .cpd files from the storage folder...........
                             getFiles(resourcePath, storageInput , objectToLabel, model_files_all, ".pcd",dataset_name);

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
                 saveDatasets (descriptors_all_train, descriptors_all_test, feat,dataset_name,split_name,savePathToOutput);


       }   else{ std::cout<<"Please select databese name (IAI or WU or BOTH)"<<std::endl;}


      // To save the string class labels in type double in folder rs_learning/data
         saveObjectToLabels(objectToClassLabelMap,feat,dataset_name, split_name,savePathToOutput);




       std::cout<<"Descriptors calculation is done"<<std::endl;


    return 0;
}  
