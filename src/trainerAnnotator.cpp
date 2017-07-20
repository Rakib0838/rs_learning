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
#include<boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>


using namespace uima;


class trainerAnnotator : public Annotator
{
private:

  // classifier type should be rssvm, rsrf, rsgbt, rsknn........
  std::string classifier_type;

  //the name of train matrix file in folder /rs_resources/objects_dataset/extractedFeat
  std::string train_data_name;

   //the name of train label matrix file in folder /rs_resources/objects_dataset/extractedFeat
  std::string train_label_name;

  // the name of trained model file, which will be generated in folder rs_learning/data
  std::string trained_model_name;

public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

     ctx.extractValue("classifier_type", classifier_type);
     ctx.extractValue("train_data_name", train_data_name);
      ctx.extractValue("train_label_name", train_label_name);

      outInfo("Name of the loaded files for classifier trainning are:"<<std::endl);

        outInfo("classifier_type:"<<classifier_type<<std::endl);
        outInfo("train_data_name:"<<train_data_name<<std::endl);
        outInfo("train_label_name:"<<train_label_name<<std::endl);


        //vector to hold split trained_model_name
        std::vector<std::string> split;
        boost::split(split, train_data_name, boost::is_any_of("_"));

      trained_model_name= split[0]+'_'+split[1]+'_'+classifier_type+"Model"+'_'+split[3];
      outInfo("trained_model_name:"<<trained_model_name<< "  will be generated in rs_learning/trainedData");

     if(classifier_type=="rssvm"){
          RSClassifier* svmObject= new RSSVM;
          outInfo("Training with SVM is going on .......");
          svmObject->trainModel(train_data_name ,train_label_name, trained_model_name);
     }
     else if(classifier_type=="rsrf"){
         RSClassifier* rfObject= new RSRF;
         outInfo("Training with RSRF is going on .......");
         rfObject->trainModel(train_data_name ,train_label_name, trained_model_name);

    }
     else if(classifier_type=="rsgbt"){
         RSClassifier* gbtObject= new RSGBT;
         outInfo("Training with RSGBT is going on .......");
         gbtObject->trainModel(train_data_name ,train_label_name, trained_model_name);

    }
    else if(classifier_type=="rsknn"){
         RSClassifier* knnObject= new RSKNN;
         outInfo("Training with RSKNN is going on .......");
         knnObject->trainModel(train_data_name ,train_label_name, trained_model_name);

    }
     else {
         outInfo("Please sellect the correct classifier_type, which is either rssvm, rsrf, rsgbt, rsknn");
     }

     outInfo("Classifier training is Done !!!");
       return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(trainerAnnotator)
