#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>

#include <rs_learning/RSClassifier.h>
#include <rs_learning/RSSVM.h>
#include <rs_learning/RSRF.h>
#include <rs_learning/RSGBT.h>
#include <rs_learning/RSKNN.h>
#include <rs/DrawingAnnotator.h>

using namespace uima;


class classifyOfflineAnnotator : public Annotator
{
private:

  // classifier type should be rssvm, rsrf, rsgbt, rsknn
  std::string classifier_type;

  // the name of trained model file in folder rs_learning/data/trainedData
  std::string trained_model_name;

  //the name of test matrix in folder rs_learning/data
  std::string test_data_name;

   //the name of test label matrix in folder rs_learning/data
  std::string test_label_name;

  //the name of actual_class_label map file in folder rs_learning/data
 std::string actual_class_label ;
 std::string train_mat;

 //To work with knn classifier........
 std::string trainData_matrix;
 std::string trainLabel_matrix;


public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

    outInfo("Name of the loaded files are:"<<std::endl);

    ctx.extractValue("classifier_type", classifier_type);
    outInfo("classifier_type:"<<classifier_type<<std::endl);

    ctx.extractValue("test_data_name", test_data_name);
    outInfo("test_data_name:"<<test_data_name<<std::endl);

    ctx.extractValue("test_label_name", test_label_name);
     outInfo("test_label_name:"<<test_label_name<<std::endl);

     ctx.extractValue("actual_class_label", actual_class_label);
     outInfo("actual_class_label:"<<actual_class_label<<std::endl);

     if( classifier_type != "rsknn"){
           ctx.extractValue("trained_model_name", trained_model_name);
           outInfo("trained_model_name:"<<trained_model_name<<std::endl);
      }


       if(classifier_type=="rssvm"){

            RSClassifier* svmObject= new RSSVM;
            outInfo("Classify with SVM is going on .......");
            svmObject->classify(trained_model_name ,test_data_name ,test_label_name,actual_class_label);
       }
       else if(classifier_type=="rsrf"){
           RSClassifier* rfObject= new RSRF;
           outInfo("Classify with RSRF is going on .......");
           rfObject->classify(trained_model_name ,test_data_name ,test_label_name,actual_class_label);

      }
       else if(classifier_type=="rsgbt"){
           RSClassifier* gbtObject= new RSGBT;
           outInfo("Classify with RGBT is going on .......");
           gbtObject->classify(trained_model_name ,test_data_name ,test_label_name,actual_class_label);

      }
       else if(classifier_type =="rsknn"){
           ctx.extractValue("trainData_matrix", trainData_matrix);
           outInfo("trainData_matrix:"<<trainData_matrix<<std::endl);

           ctx.extractValue("trainLabel_matrix", trainLabel_matrix);
           outInfo("trainLabel_matrix:"<< trainLabel_matrix<<std::endl);

           RSKNN* knnObject= new RSKNN;
           outInfo("Classify with RSKNN is going on .......");
           knnObject->classifyKNN(trainData_matrix ,trainLabel_matrix,test_data_name ,test_label_name,actual_class_label);

      }

       else {
           outInfo("Please sellect the correct classifier_type, which is either rssvm, rsrf, rsgbt, rsknn");
       }


    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }


};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(classifyOfflineAnnotator)
