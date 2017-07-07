# rs_learning
rs_learning is a Robosherlock package as well as ROS package.

# Prerequisite: Robosherlock, caffe, PCL, openCV

# The rs_learning package consists of three modules. 
   1. Module for extracting features.
   2. Module for creating trainedModel
      for different classifiers.
   3. Module for classifying images.
# Extracting feature module:
# Usage:
       ### To get the help
          rosrun rs_learning -h
       
       ### To extract feature
          rosrun rs_learning -f file -s storage -d database_name -o split -r feat
   
         where:
                      file: It is a .ymal file, contains informations about objects and object's class
                            label. The file should be in catkin workspace rs_resources/objects_dataset folder.
            
                   storage: It is the name of the folder of object database. 
                            The folder should be in catkin workspace rs_resources/objects_dataset.
           
             database_name: It should be IAI (to use database from Institue for atificial inteligent) or
                            WU (to use database from Washington University). So please pay attention when
                            select the storage folder name.
                     
                     split: It should be INS or ALL (for IAI ) and ONE (for WU).
                             
                            ALL: It stands for calculating features of all images in a database.
                            
                            INS: It stands for calulating features and seperate very fourth image 
                                 data as test data and all the rest as train data.
                            
                            ONE: It splits every first object of a class as test data and all the 
                                 rest as train data.
                     
                      feat: It should be CNN or VGG16 (RGB data) and VFH or CVFH (for RGB-D data).

            The above command should generate following files in rs_learning/data folder. So check
            data folder is there, if not create one and name it as data. 
                          
                          1. database_name_feat_split_ClassLabelfile.txt 
                          
                          2. database_name_feat_split_MatTrainfile.yaml
                          
                          3. database_name_feat_split_MatTrainLabelfile.yaml
                          
                          4. database_name_feat_split_MatTestfile.yaml
                      
                          5. database_name_feat_split_MatTestLabelfile.yaml
                  
                    if someone selects split parameter as ALL the above first three files will be generated. 
             
     ##Example: If someone type the following command: 
                        
         rosrun rs_learning -f ObjectOur -s partial_views -d IAI -o INS -r CNN
                      
         output should be following:
        
                                              
                          1. IAI_CNN_INS_ClassLabelObjectOur.txt
                          
                          2. IAI_CNN_INS_MatTrainObjectOur.yaml
                          
                          3. IAI_CNN_INS_MatTrainLabelObjectOur.yaml
                          
                          4. IAI_CNN_INS_MatTestObjectOur.yaml
                      
                          5. IAI_CNN_INS_MatTestLabelObjectOur.yaml

      In Robosherlock each annotator has one .xml file in Descriptors/annotators folder. 
      And the ensemble of annotators is called analysis engine.
   
# TrainedModel creator module:
    If someone wants to create the TrainedModel of data for specific classifer,
    should first provide the following parameter's value in trainerAnnotator.xml
    file. It shoulde be genarated a TrainedModel file as database_name_feat_split_MatTrainfile.yaml
    in rs_learning/data/trainedData folder. 
                     
                      1. classifier_type: It should be rssvm (for support vector mechine) or
                                          rsrf (for random forest) or rsgbt (for gradient boost tree) or
                                          rsknn (for k-Nearest neighbour) .

                      2. train_data_name: The name of the train data file (database_name_feat_split_MatTrainfile) 
                                          in folder rs_learning/data
                      
                      3. train_label_name: The name of the data trainLabel file 
                         (database_name_feat_split_MatTrainLabelfile) in folder rs_learning/data 
                
     ## Example: If someone choose parameters classifier_type as rssvm, train_data_name
                            as IAI_CNN_INS_MatTrainObjectOur and train_label_name name as 
                            IAI_CNN_INS_MatTrainLabelObjectOur in trainerAnnotator.xml file and type the
                            following command on terminal.
                          
                            rosrun robosherlock run model_trainer
  
               then as output IAI_CNN_INS_rssvmModel_ObjectOur should be generated 
               in rs_learning/data/trainedData folder. 
                        
#########################################################################                       
  # Classify Image Module: 
  It is divided into two parts classify offline and online. 
  If someone has test data on hand, he can use classify_offline 
  annotator and classifies the images. The command for that:
                          
      rosrun robosherlock run classify_offline

      Before enter the command please tune the following parameter in classifyOfflineAnnotator.xml file.

         1.classifier_type: It should be rssvm or rsrf or rsgbt or rsknn
  
         2.trained_model_name: The name of the trainedModel file (Ex. if someone selects classifier_type
                              (= rssvm),  then traindModel should look like IAI_CNN_INS_rssvmModel_ObjectOur).
                         
         3.test_data_name: It should be the test data file name (Ex.IAI_CNN_INS_MatTestObjectOur.yaml)
                         
         4. test_label_name: The name of the testLabel data file (Ex.IAI_CNN_INS_MatTestLabelObjectOur)
                         
         5. actual_class_label: The name of classLabel file (Ex.IAI_CNN_INS_ClassLabelObjectOur)
                       

      If the classifier_type (=rsknn), instead of trained_model_name selects the following two files.
                          
                  1. trainDatamatrix: The name of the train matrix file (Ex.IAI_CNN_INS_MatTrainObjectOur)
                         
                  2. trainlabel_matrix: The name of the trainLabel matrix file (IAI_CNN_INS_MatTrainLabelObjectOur)
                          

 
       if test data is coming from a .bag file or any databese or from real time robot manipulation
       task, the process is called online. Then someone has to use the following command.

                   rosrun robosherlock run my_demo
                      
       my_demo is an analysis engine with many Annotators(specially classifiers). Each classifier 
       has two options. It can classify or set the groundtruth for the images.So before runing the 
       above command please tune the parameters in the respective annotator's .xml file.
       The parameters name are same for classifiers (rssvm, rsrf, rsgbt) and they are:
                           
              1. set_mode: It should be CL (to classify) and GT (to set groundtruth )                           
                                   
              2. trained_model_name: name of the trainedModel (Ex.IAI_CNN_INS_rssvmModel_ObjectOur).
                         
              3. actual_class_label: name of classLabel file (Ex.IAI_CNN_INS_ClassLabelObjectOur)
    
                       
       And for classifier (rsknn), please selects set_mode (=rsknn) and instead of parameter 
       (trained_model_name) tune the the following parameters.
                       
               trainKNN_matrix: The name of the train matrix file (Ex.IAI_CNN_INS_MatTrainObjectOur)
                           
               trainKNNLabel_matrix: The name of the trainLabel matrix file (IAI_CNN_INS_MatTrainLabelObjectOur) 
   
       # Attention: When classify images online please make sure that the image features coming 
                   from Robosherlock annotators(Ex. PCLfeatureExtractor or caffe ) must be the 
                   same as the respective trainedModel's features.
 
 
 
