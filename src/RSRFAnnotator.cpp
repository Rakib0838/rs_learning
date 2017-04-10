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
using namespace uima;

class RSRFAnnotator : public Annotator
{
private:
  float test_param;

public:

  RSRF tui;
  RSClassifier* pon= &tui;


  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

    /* To train the classifier, call the function (trainModel), which takes train data from folder /rs_learning/data.
     The function takes Mat_train_ , label_train_ as parameters and produces a trained mat file as the name given as it's
     third parameter in folder rs_learning/data/trainedData */

   // pon->trainModel("Mat_train_OBJ_CNN" ,"label_train_OBJ_CNN","trained_saved_RF_OBJ_CNN");

    /* If the classifier is already trained on need to call  the above function (trainModel). The function (Classify) takes
      trained data (in folder rs_learning/data/trainedData ) , Mat_test_ and it's labels (in folder rs_learning/data) as inputs
      and show the results of the classifications */

   pon->classify("trained_saved_RF_OBJ_CNN" ,"Mat_test_OBJ_CNN" ,"label_test_OBJ_CNN","ClassLabel_OBJ_CNN");

    ctx.extractValue("test_param", test_param);
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {




   
 /*
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
    outInfo("Test param =  " << test_param);
    cas.get(VIEW_CLOUD,*cloud_ptr);

    outInfo("Cloud size: " << cloud_ptr->points.size());
    outInfo("took: " << clock.getTime() << " ms.");   */
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(RSRFAnnotator)
