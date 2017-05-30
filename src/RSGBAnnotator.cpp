
/*

#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs_learning/RSClassifier.h>
#include <rs_learning/RSGBT.h>
#include <rs/DrawingAnnotator.h>

using namespace uima;


class RSGBAnnotator : public DrawingAnnotator
{
private:
  float test_param;

public:
   // Incomplete work...........Gradient boost trees....
      RSGBT rakiba;
       RSGBT* objGBT= &rakiba;

       RSGBAnnotator(): DrawingAnnotator(__func__)
       {

       }

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("test_param", test_param);

      std::cout<<"sdsajfhiahisgyiashighig";
 // objGBT->trainModel("Mat_train_ONE_CNN" , "label_train_ONE_CNN", "trained_GBT_CNN__ONE_WU_51");

//  objGBT->classify("trained_GBT_CNN__ONE_WU_51" , "Mat_test_ONE_CNN" , "label_test_ONE_CNN", "ClassLabel_ONE_CNN");

     return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
    outInfo("Test param =  " << test_param);
    cas.get(VIEW_CLOUD,*cloud_ptr);

    outInfo("Cloud size: " << cloud_ptr->points.size());
    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(RSGBAnnotator)

*/
