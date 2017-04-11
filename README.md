# rs_learning

..................To extract the feature..................
- rosrun rs_learning featureExtractor -h  // show the help option
- with -f enter the file name where all the object's names are listed.
- with -s enter the object storage folder name (eg. partial_views)
- with -o select ALL ,which calculate all features.
  OBJ ,which create train and test sets of the data , where 
  every third object of the sub-class consider as test data.
  INS ,which create train and test sets of the dataset, where every forth (image or cloud)
  f the hole datasets considered as the element of the test dataset. 
- with -r CNN or -r VFH feature can be selected.
All the produced datas can be founded in rs_learning/data folder.

....................To train the classifier...............<br />
 Call the function (trainModel), which takes train data from folder /rs_learning/data.
 The function takes Mat_train_ , label_train_ as parameters and produces a trained mat file as the name given as it's
 third parameter in folder rs_learning/data/trainedData. 
 
 ..........................To classify....................<br />
 If the classifier is already trained on need to call  the above function (trainModel). 
 The function (Classify) takes trained data (in folder rs_learning/data/trainedData ),
 Mat_test_, it's labels and classLabel (in folder rs_learning/data) as inputs
 and show the results of the classifications 
 
 please also check if all the required annotators are listed in my_demo (aggregate analysis engine)
