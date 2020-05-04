cmake_minimum_required(VERSION 3.6)
set(KEYWORD objectDetector)
set(NAME_VENV ${KEYWORD}_venv)
project($KEYWORD)

set(PROGRAM_PERMISSIONS_DEFAULT
    OWNER_WRITE OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

# Find Python and Virtualenv.
find_program(VIRTUALENV virtualenv)

if(NOT VIRTUALENV)
    message(FATAL_ERROR "Could not find `virtualenv` in PATH")
endif()

set(VIRTUALENV ${VIRTUALENV})

# Find Yarp and iCubCONTRIB packages
find_package(YARP  REQUIRED)
FIND_PACKAGE(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)

icubcontrib_set_default_prefix()
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")


# Generate the virtualenv and ensure it's up to date.
add_custom_command(
    OUTPUT venv
    COMMAND ${VIRTUALENV} ${NAME_VENV}
)
add_custom_command(
    OUTPUT venv.stamp
    DEPENDS venv requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt requirements.txt
    COMMAND ./${NAME_VENV}/bin/pip install -r requirements.txt
)

# Generate the binary using pyinstaller and put the binary in the bin folder
add_custom_command(
    OUTPUT build_binary
    COMMAND ./${NAME_VENV}/bin/pyinstaller  ${CMAKE_CURRENT_SOURCE_DIR}/src/objectDetector.spec --distpath bin

)


# Custom targets for CMake
# Create a virutal-env and install all requirements
add_custom_target(install_requirements ALL
    DEPENDS venv.stamp 
    SOURCES requirements.txt
)

# Create the binary file using pyinstaller and install it
add_custom_target(build_exe ALL
    DEPENDS build_binary
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/${KEYWORD} PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION bin)

add_subdirectory(app)
