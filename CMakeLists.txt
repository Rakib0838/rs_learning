cmake_minimum_required(VERSION 2.8.3)
project(rs_learning)
find_package(catkin REQUIRED robosherlock)
################################################################################
## Constants for project                                                      ##
################################################################################
set(NAMESPACE rs_learning)
set(TYPESYSTEM_CPP_PATH ${PROJECT_SOURCE_DIR}/include/rs_learning/types)
set(TYPESYSTEM_XML_PATH ${PROJECT_SOURCE_DIR}/descriptors/typesystem)
set(ANNOTATOR_PATH      ${PROJECT_SOURCE_DIR}/descriptors/annotators)
set(ENGINE_PATH         ${PROJECT_SOURCE_DIR}/descriptors/analysis_engines)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake)
################################################################################
## Configure project configuration files                                      ##
################################################################################
configure_file(${RS_PROJECT_CONFIG} ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake @ONLY)
################################################################################
## Find all include directories                                               ##
################################################################################
find_include_dirs(RS_INCLUDE_DIRS_LIST)
catkin_package(
   CFG_EXTRAS ${PROJECT_NAME}_config.cmake
   )
################################################################################
## Package dependencies                                                       ##
################################################################################
#Add required catkin packages here
#find_package(catkin REQUIRED COMPONENTS xxx)
#Add required system packages here
#find_package(xxx)
################################################################################
## Build                                                                      ##
################################################################################
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${RS_INCLUDE_DIRS_LIST}
        ${catkin_INCLUDE_DIRS}
        )
################################################################################
## Update analysis engines, typesystem and include all relevant files         ##
################################################################################
##Update xml list of annotators inside analysis engines
update_analysis_engines(robosherlock)
## generate classes from the typesystem xml files
generate_type_system(robosherlock)
#find all relevant files
find_additional_files()
################################################################################
## Sub Projects                                                               ##
################################################################################
#If you want to divide your projects into subprojects include the subdirectories
#each containing a CMakeLists.txt here
#add_subdirectory(src/xxx)

rs_add_library(rs_rSSvmAnnotator src/RSSvmAnnotator.cpp
                               src/RSClassifier.cpp 
                               src/RSSVM.cpp)

target_link_libraries(rs_rSSvmAnnotator ${CATKIN_LIBRARIES}
                                       ${OpenCV_LIBRARIES}
                                        ${Boost_LIBRARIES}
                                        ${PCL_LIBRARIES})




rs_add_library(rs_rSRFAnnotator src/RSRFAnnotator.cpp
                               src/RSClassifier.cpp
                               src/RSRF.cpp )
target_link_libraries(rs_rSRFAnnotator ${CATKIN_LIBRARIES}
                                       ${OpenCV_LIBRARIES}
                                        ${Boost_LIBRARIES}
                                        ${PCL_LIBRARIES})

rs_add_library(rs_rSGBAnnotator src/RSGBAnnotator.cpp
                               src/RSClassifier.cpp 
                               src/RSGBT.cpp)

target_link_libraries(rs_rSGBAnnotator ${CATKIN_LIBRARIES}
                                       ${OpenCV_LIBRARIES}
                                        ${Boost_LIBRARIES}
                                        ${PCL_LIBRARIES})



add_executable(featureExtractor src/ext/featureExtractor.cpp)
target_link_libraries(featureExtractor ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${Caffe_LIBRARIES})
 

 #add_library(rs_classifier src/rs_classifier.cpp)
  #target_link_libraries(rs_classifier ${catkin_LIBRARIES} ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES}  ${PCL_LIBRARIES})



#add_executable(main src/main.cpp)
#target_link_libraries(main ${CATKIN_LIBRARIES} 
#                          ${OpenCV_LIBRARIES}
 #                         ${Boost_LIBRARIES}
  #                         ${PCL_LIBRARIES})

