# rs_learning
   rs_learning is a Robosherlock package as well as ROS package.

  # Prerequisite: Robosherlock, caffe, PCL, openCV, rs_resources

  # The rs_learning package consists of three modules. 
   1. Module for extracting features.
   2. Module for creating trainedModel
      for different classifiers.
   3. Module for classifying images.
  # Extracting feature module:
  # Usage:
       ### To get the help
          rosrun rs_learning featureExtractor -h
       
       ### To extract feature
          rosrun rs_learning featureExtractor -s splitName -i storage -d datasetName -f feat
   
         where:
                splitName: It is a .ymal file, contains informations about objects and object's class
                            label. The file should be in catkin workspace rs_resources/objects_dataset/splits folder.
            
                   storage: It is the name of input image storage folder.The folder should be in 
                            rs_resources/objects_dataset. For this project we use two datasets, 
                            one is  kitchen environmemts dataset from Institue for atificial inteligent 
                            and the other one is from Washington University dataset. Parameter's (storage) value 
                            should be iaiImageFolder/wuImageFolder if someone wants to use both datasets at once
                            else just iaiImageFolder or wuImageFolder folder name.
                             
           
             datasetName: It's value should be should be IAI (to use dataset from Institue for 
                            atificial inteligent) and WU (to use dataset from Washington University).
                            If someone wants to use both datasets at once, he should select the 
                            parameter's value as BOTH. 
                           
                     
                      feat: It should be CNN or VGG16 (RGB data) and VFH or CVFH (for RGB-D data).

            The above command should generate following files in rs_learning/data folder. So check
            data folder is there, if not create one and name it as data. 
                          
                          1. datasetName_feat_ClassLabel_splitName.txt 
                          
                          2. datasetName_feat_MatTrain_splitName.yaml
                          
                          3. datasetName_feat_MatTrainLabel_splitName.yaml
                          
                          4. datasetName_feat_MatTest_splitName.yaml
                      
                          5. datasetName_feat_MatTestLabel_splitName.yaml
                  
             
     ##Example: If someone type the following command: 
                        
         rosrun rs_learning featureExtractor -s ObjectOur -i partial_views -d IAI -f CNN
                      
         Output should be following:
        
                                              
                          1. IAI_CNN_ClassLabel_ObjectOur.txt
                          
                          2. IAI_CNN_MatTrain_ObjectOur.yaml
                          
                          3. IAI_CNN_MatTrainLabel_ObjectOur.yaml
                          
                          4. IAI_CNN_MatTest_ObjectOur.yaml
                      
                          5. IAI_CNN_MatTestLabel_ObjectOur.yaml

      In Robosherlock each annotator has one .xml file in Descriptors/annotators folder. 
      And the ensemble of annotators is called analysis engine.
   
  # TrainedModel creator module:
    If someone wants to create the TrainedModel of the data for specific classifer,
    should first provide the following parameter's value in trainerAnnotator.xml
    file. It will genarate a TrainedModel file as datasetName_feat_classifier_typeModel_ObjectOur.yaml
    in rs_learning/data/trainedData folder. 
                     
                      1. classifier_type: It's value should be rssvm (for support vector mechine) or
                                          rsrf (for random forest) or rsgbt (for gradient boost tree) or
                                          rsknn (for k-Nearest neighbour) .

                      2. train_data_name: The name of the train data file (datasetName_feat_MatTrain_ObjectOur) 
                                          in path rs_resources/objects_dataset/extracetedFeat
                      
                      3. train_label_name: The name of the data trainLabel file 
                        (datsetName_feat_MatTrainLabel_splitName) in path 
                         rs_resources/objects_dataset/extracetedFeat/ 
                
      ## Example: If someone choose parameters classifier_type as rssvm, train_data_name
                            as IAI_CNN_MatTrain_ObjectOur and train_label_name name as 
                            IAI_CNN_MatTrainLabel_ObjectOur in trainerAnnotator.xml file and type the
                            following command on terminal.
                          
                            rosrun robosherlock run model_trainer
  
               then as output IAI_CNN_rssvmModel_ObjectOur should be generated 
               in rs_learning/trainedData folder. 
                        
#########################################################################                       
   # Classify Image Module: 
  It is divided into two parts classify offline and online. 
  If someone has test data on hand, he can use classify_offline 
  annotator and classifies the images. The command for that:
                          
      rosrun robosherlock run classify_offline

      Before enter the command please tune the following parameter in classifyOfflineAnnotator.xml file.

         1.classifier_type: It should be rssvm or rsrf or rsgbt or rsknn
  
         2.trained_model_name: The name of the trainedModel file (Ex. if someone selects classifier_type
                              (= rssvm),  then traindModel should look like IAI_CNN_rssvmModel_ObjectOur).
                         
         3.test_data_name: It should be the test data file name (Ex.IAI_CNN_MatTest _ObjectOur.yaml)
                         
         4. test_label_name: The name of the testLabel data file (Ex.IAI_CNN_MatTestLabel_ObjectOur)
                         
         5. actual_class_label: The name of classLabel file (Ex.IAI_CNN_ClassLabel_ObjectOur)
                       

      If the classifier_type (=rsknn), instead of trained_model_name selects the following two files.
                          
                  1. trainDatamatrix: The name of the train matrix file (Ex.IAI_CNN_MatTrain_ObjectOur)
                         
                  2. trainlabel_matrix: The name of the trainLabel matrix file (IAI_CNN_MatTrainLabel_ObjectOur)
                          

 
       If test data is coming from a .bag file or any databese or from real time robot manipulation
       task, the process is called online. Then someone has to use the following command.

                   rosrun robosherlock run my_demo
                      
       my_demo is an analysis engine with many Annotators(specially classifiers). Each classifier 
       has two options. It can classify or set the groundtruth for the images.So before runing the 
       above command please tune the parameters in the respective annotator's .xml file.
       The parameters name are same for classifiers (rssvm, rsrf, rsgbt) and they are:
                           
              1. set_mode: It should be CL (to classify) and GT (to set groundtruth )                           
                                   
              2. trained_model_name: name of the trainedModel (Ex.IAI_CNN_rssvmModel_ObjectOur).
                         
              3. actual_class_label: name of classLabel file (Ex.IAI_CNN_ClassLabel_ObjectOur)
    
                       
       And for classifier (rsknn), please selects set_mode (=rsknn) and instead of parameter 
       (trained_model_name) tune the the following parameters.
                       
               trainKNN_matrix: The name of the train matrix file (Ex.IAI_CNN_MatTrain_ObjectOur)
                           
               trainKNNLabel_matrix: The name of the trainLabel matrix file (IAI_CNN_MatTrainLabel_ObjectOur) 
   
       # Attention: When classify images online please make sure that the image features coming 
                   from Robosherlock annotators pipeline (Ex. PCLfeatureExtractor or caffe ) must be the 
                   same as the respective trainedModel's features.
 
 
 
