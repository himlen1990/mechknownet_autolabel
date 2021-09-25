cmake_minimum_required(VERSION 2.8.3)
project(mechknownet_autolabel)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
add_compile_options( -std=c++11 )

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  roslib
  rospy
  image_transport
  std_msgs
  sensor_msgs
  visualization_msgs
  tf
  pcl_ros
  )


find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )


add_service_files(
  FILES
  mrcnn.srv
  )


generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   sensor_msgs
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
CATKIN_DEPENDS roscpp message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a C++ library
# add_library(mechknownet_autolabel
#   src/${PROJECT_NAME}/mechknownet_autolabel.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(mechknownet_autolabel ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(mechknownet_autolabel_node src/mechknownet_autolabel_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(mechknownet_autolabel_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(mechknownet_autolabel_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS mechknownet_autolabel mechknownet_autolabel_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_mechknownet_autolabel.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

add_executable(function_demonstration_activated src/function_demonstration_activated.cpp)
target_link_libraries(function_demonstration_activated ${OpenCV_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES})

add_executable(function_demonstration_associated src/function_demonstration_associated.cpp)
target_link_libraries(function_demonstration_associated ${OpenCV_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES})

#add_executable(trajectory_demonstration src/trajectory_demonstration.cpp)
#target_link_libraries(trajectory_demonstration ${OpenCV_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES})

add_executable(trajectory_demonstration_tabletop src/trajectory_demonstration_tabletop.cpp)
target_link_libraries(trajectory_demonstration_tabletop ${OpenCV_LIBS} ${PCL_LIBRARIES}  ${catkin_LIBRARIES})

add_executable(coordinate_converter src/coordinate_converter.cpp)
target_link_libraries(coordinate_converter ${PCL_LIBRARIES}  ${catkin_LIBRARIES})

#add_executable(grasping_demonstration src/grasping_demonstration.cpp)
#target_link_libraries(grasping_demonstration ${PCL_LIBRARIES}  ${catkin_LIBRARIES})
