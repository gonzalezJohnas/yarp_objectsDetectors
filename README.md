# yarp_objectsDetectors
Yarp wrapper to run object detection models (pb graph) and CMake support to create  a single binary file 

## Yarp python module
    Parameters:
    label_path -> Absolute path to the protobuf text file containing the labels ( ex app/conf/face_label.pbtxt)
    model_path -> Absolute path of a frozen protobuf graph ( ex app/conf/face_inference_graph.pb)
    width -> image width of the input image ( default 320)
    height -> image height of the input image ( default 240)
    
## Generating the binary

    To generate the binary just follow the classical CMake procedure
    - Create a build folder
    - run ccmake/cmake
    - make install
    
## System Dependencies 
- Yarp with python binding
- iCubCONTRIB (optional)

## Python dependency 
They are listed in the requirements.txt