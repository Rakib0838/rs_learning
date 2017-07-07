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
#include <rs_learning/RSSVM.h>



//....................................Support vector machine........................................

RSSVM::RSSVM()
{

}

// To train the SVM..................................................
 void RSSVM:: trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)
 {
        cv::Mat train_matrix;
        cv::Mat train_label;
        readDescriptorAndLabel(train_matrix_name, train_label_name,train_matrix,train_label);
               std::cout<<"size of train matrix:"<<train_matrix.size()<<std::endl;
               std::cout<<"size of train label:"<<train_label.size()<<std::endl;



      // Set optimization parameters......................................

               CvSVMParams params = CvSVMParams(
                                            CvSVM::C_SVC,   // Type of SVM, here N classes
                                            CvSVM::LINEAR,  // kernel type
                                            0.0,			// kernel parameter (degree) for poly kernel only
                                            0.0,			// kernel parameter (gamma) for poly/rbf kernel only
                                            0.0,			// kernel parameter (coef0) for poly/sigmoid kernel only
                                            2,				// SVM optimization parameter C
                                            0,				// SVM optimization parameter nu
                                            0,				// SVM optimization parameter p
                                            NULL,			// class wieghts (or priors)
    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001) // termination criteria
                                                );
     

      // Train SVM classifier......................................

        CvSVM* my_svm =new CvSVM;
       my_svm->train_auto(train_matrix, train_label, cv::Mat(), cv::Mat(), params, 10);
     //   my_svm->train(train_matrix, train_label, cv::Mat(), cv::Mat(), params);


        // To save the trained data.............................

        my_svm->save((saveOrLoadTrained(trained_file_name)).c_str());
 }


// To predict the class using SVM................................

 void RSSVM:: classify (std::string trained_file_name_saved,
               std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{

    cv::Mat test_matrix;
    cv::Mat test_label;

   // To load the test data and it's label.............................

  readDescriptorAndLabel(test_matrix_name, test_label_name, test_matrix, test_label);
  std::cout<<"size of test matrix :"<<test_matrix.size()<<std::endl;
  std::cout<<"size of test label"<<test_label.size()<<std::endl;

    CvSVM* your_svm=new CvSVM ;

    // To load the trained data..............................
    your_svm->load((saveOrLoadTrained(trained_file_name_saved)).c_str());

    // To count the support vector................................
      int in= your_svm->get_support_vector_count();
     std::cout<<"The number of support vector:"<<in<<std::endl;



    //convert test label matrix into a vector
     std::vector<double> con_test_label;
      test_label.col(0).copyTo(con_test_label);

    // Container to hold the integer value of labels............................
            std::vector<int> actual_label;
            std::vector<int> predicted_label;

    //Loop to prdict the rsult............................
       for(int i=0; i< test_label.rows; i++) {
            double res = your_svm->predict(test_matrix.row(i));
            int prediction= res;
             predicted_label.push_back(prediction);
            double lab = con_test_label[i];
            int actual_convert= lab;
            actual_label.push_back(actual_convert);

          std::cout<<"actuall class is:"<<actual_convert<<"::"<<"prediction is :"<< prediction<<std::endl;
         //   std::cout<<"prediction class is::"<<prediction <<std::endl;
          }


   std::cout<<"result of SVM :"<<std::endl;

   evaluation(actual_label, predicted_label, obj_classInDouble );

}


 // To predict the class using SVM................................

  void RSSVM:: classify_own_svm (std::string trained_file_name_saved, cv::Mat test_mat, double &clsLab)
 {
      std::cout<<"size of test matrix :"<<test_mat.size()<<std::endl;

     CvSVM* your_svm=new CvSVM ;

     // To load the trained data..............................
     your_svm->load((saveOrLoadTrained(trained_file_name_saved)).c_str());

     // To count the support vector................................
   //    int in= your_svm->get_support_vector_count();
   //   std::cout<<"The number of support vector:"<<in<<std::endl;




             double res = your_svm->predict(test_mat);

             std::cout<<"prediction class is::"<<res<<std::endl;
                       clsLab =res;



 }

  void RSSVM::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det)
   {

      // To load the test data and it's label.............................

     std::cout<<"size of test matrix :"<<test_mat.size()<<std::endl;


       CvSVM* your_svm=new CvSVM ;

       // To load the trained data..............................
       your_svm->load((saveOrLoadTrained(trained_file_name_saved)).c_str());

       // To count the support vector................................
         int in= your_svm->get_support_vector_count();
        std::cout<<"The number of support vector:"<<in<<std::endl;


       //Loop to prdict the rsult............................

               double res = your_svm->predict(test_mat);
               det =res;

               std::cout<<"prediction class is :"<< res<<std::endl;

   }




  void RSSVM::RsAnnotation (uima::CAS &tcas,std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster,std::string set_mode)
  {

      rs::Classification classResult= rs::create<rs::Classification>(tcas);
      classResult.classname.set(class_name);
      classResult.classifier("Support Vector Machine");
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



 RSSVM::~RSSVM()
 {

 }

