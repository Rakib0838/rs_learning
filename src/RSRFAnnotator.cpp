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

  std::string test_param_rf;
  cv::Mat color;
   std::vector<std::string> model_labels;

public:

  RSRFAnnotator(): DrawingAnnotator(__func__){

 }

  RSRF rsrfInstance;
  RSClassifier* rfObject= &rsrfInstance;


  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

    ctx.extractValue("test_param_rf", test_param_rf);

  // read the class label from folder rs_learning/data....................................

   // To call the object class label
    if(test_param_rf =="instant")
    {
        rfObject->setLabels("class_label", model_labels);
     }
   else if(test_param_rf =="shape")
     {
   //To call the shape class label.
   rfObject->setLabels("class_label_shape", model_labels);
     }


    /* To train the classifier, call the function (trainModel), which takes train data from folder /rs_learning/data.
     The function takes Mat_train_ , label_train_ as parameters and produces a trained mat file as the name given as it's
     third parameter in folder rs_learning/data/trainedData */

      //  rfObject->trainModel("Mat_train_ONE_CNN" , "label_train_ONE_CNN", "trained_RF_ONE_CNN_WU");

    /* If the classifier is already trained on need to call  the above function (trainModel). The function (Classify) takes
      trained data (in folder rs_learning/data/trainedData ) , Mat_test_ and it's labels (in folder rs_learning/data) as inputs
      and show the results of the classifications */

     // rfObject->classify("trained_RF_ONE_CNN_WU" ,"Mat_test_ONE_CNN" ,"label_test_ONE_CNN","ClassLabel_ONE_CNN");


    return UIMA_ERR_NONE;
  }



  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }



  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {

      rs::SceneCas cas(tcas);
      rs::Scene scene = cas.getScene();

      cas.get(VIEW_COLOR_IMAGE_HD, color);
      std::vector<rs::Cluster> clusters;
      scene.identifiables.filter(clusters);

      if(test_param_rf == "instant")
      {
          // To work with caffe feature...............
          rfObject->processCaffeFeature("trained_RF_ALL_CNN_OUR",clusters, rfObject, color, model_labels);
       }
     else if(test_param_rf == "shape")
       {
          //To work with VFH feature.....................
          rfObject->processVFHFeature("trained_RF_ALL_VFH_OUR", clusters, rfObject, color, model_labels);

       }


     return UIMA_ERR_NONE;

  }



  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
  }


};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(RSRFAnnotator)
