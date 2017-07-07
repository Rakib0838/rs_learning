

#include <uima/api.hpp>
#include <iostream>
#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include<ros/package.h>
#include<boost/filesystem.hpp>

#include <rs_learning/RSClassifier.h>
#include <rs_learning/RSRF.h>

#include <rs/DrawingAnnotator.h>
using namespace uima;

class RSRFAnnotator : public DrawingAnnotator
{
private:

    cv::Mat color;

    //set_mode should be GT(groundTruth) or CF (classify) dtata which come from database....
       std::string set_mode;

      // dataset_use should be IAI (kitchen data from IAI) or WU (data from Washington University)...
      std::string dataset_use;

      // feature_use should be VFH, CVFH, CNN, VGG16 .....
      std::string feature_use;

      // the name of trained model ifle in folder rs_learning/data/trainedData
      std::string trained_model_name;

      //vector to hold split trained_model_name
      std::vector<std::string> split_model;

      //the name of actual_class_label map file in folder rs_learning/data
     std::string actual_class_label;

     //vector to hold classes name
     std::vector<std::string> model_labels;

public:

  RSRFAnnotator(): DrawingAnnotator(__func__){

 }

  RSClassifier* rfObject= new RSRF;


  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

     ctx.extractValue("set_mode", set_mode);
    ctx.extractValue("trained_model_name", trained_model_name);
     ctx.extractValue("actual_class_label", actual_class_label);

     outInfo("Name of the loaded files for RF are:"<<std::endl);

       outInfo("set_mode:"<<set_mode<<std::endl);
       outInfo("trained_model_name:"<<trained_model_name<<std::endl);
       outInfo("actual_class_label:"<<actual_class_label<<std::endl);

       rfObject->setLabels(actual_class_label, model_labels);

       boost::split(split_model, trained_model_name, boost::is_any_of("_"));

       dataset_use= split_model[0];
       outInfo("dataset_use:"<<dataset_use<<std::endl);

       feature_use= split_model[1];
       outInfo("feature_use:"<<feature_use<<std::endl);


    return UIMA_ERR_NONE;
  }



  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }



  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {

      outInfo("RSRFAnnotator is running:"<<std::endl);
       rs::SceneCas cas(tcas);
       rs::Scene scene = cas.getScene();

       cas.get(VIEW_COLOR_IMAGE_HD, color);
       std::vector<rs::Cluster> clusters;
       scene.identifiables.filter(clusters);

       if(set_mode =="CL"){


           if(dataset_use =="WU" &&  feature_use=="VFH")
           {
                std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);

           }

           else if(dataset_use =="WU" &&  feature_use=="CVFH")
           {
                std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);

           }
           else if(dataset_use=="WU" &&  feature_use=="CNN")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
              rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);
           }
           else if(dataset_use=="WU" &&  feature_use=="VGG16")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
              rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);
           }

           else if(dataset_use=="IAI" &&  feature_use=="VFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
                rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);
           }
           else if(dataset_use=="IAI" &&  feature_use=="CVFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
                rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);
           }
           else if(dataset_use=="IAI" &&  feature_use=="CNN")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }
           else if(dataset_use=="IAI" &&  feature_use=="VGG16")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }
            else{ outInfo("Please sellect the correct value of parameter(feature_use):VFH,CVFH,CNN,VGG16"<<std::endl);}

       }
       else if(set_mode =="GT")
       {
           if(dataset_use=="WU" &&  feature_use=="VFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);

           }
           else if(dataset_use=="WU" &&  feature_use=="CVFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);

           }

           else if(dataset_use =="WU" &&  feature_use =="CNN")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }
           else if(dataset_use =="WU" &&  feature_use =="VGG16")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }

           else if(dataset_use =="IAI" &&  feature_use=="VFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);
           }
           else if(dataset_use =="IAI" &&  feature_use=="CVFH")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processVFHFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color,model_labels, tcas);
           }
           else if(dataset_use =="IAI" &&  feature_use=="CNN")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }
           else if(dataset_use =="IAI" &&  feature_use=="VGG16")
           {
               std::cout<<"Calculation starts with :" <<set_mode<<"::"<<dataset_use <<"::"<<feature_use<<std::endl;
               rfObject->processCaffeFeature(trained_model_name,set_mode,dataset_use,feature_use,clusters, rfObject, color, model_labels, tcas);

           }
           else{ outInfo("Please sellect the correct value of parameter(feature_use):VFH,CVFH,CNN,VGG16"<<std::endl);}


       }
       else {
                 outInfo("Please set the parameter (set_mode) to CL or GT ");
             }



         outInfo("calculation is done with RSRF"<<std::endl);


     return UIMA_ERR_NONE;

  }



  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
  }


};


// This macro exports an entry point that is used to create the annotator.
MAKE_AE(RSRFAnnotator)
