
##### Qt specifics #####

TEMPLATE = app

CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

###### defines ########

QMAKE_CXXFLAGS_RELEASE += -O2 -DNDEBUG

######## files ########

SOURCES += \
    src/main.cpp \
    src/cudagol.cpp

HEADERS += \
    src/cudagol.h \
    src/cudautil.h

OTHER_FILES += \
    src/cudagol_kernel.cu

CUDA_SOURCES += \
    src/cudagol_kernel.cu

###### libraries #####

LIBS += -lrt


####### cuda setup ########
# http://stackoverflow.com/questions/16053038/cuda-with-qt-in-qt-creator-on-ubuntu-12-04

NVCC_OPTIONS = --use_fast_math

CUDA_LIBS = -lcuda -lcudart

CUDA_SDK = "/usr/lib/nvidia-cuda-toolkit/"   # Path to cuda SDK install
CUDA_DIR = "/usr/lib/nvidia-cuda-toolkit/"   # Path to cuda toolkit install

SYSTEM_NAME = unix          # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'

# add cuda include paths
INCLUDEPATH += $$CUDA_DIR/include

# add cuda library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/

CUDA_OBJECTS_DIR = ./

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
