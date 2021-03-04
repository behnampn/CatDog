HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = miopen_conv_parallel.cpp
OBJECTS = $(SOURCES:.cpp=.o)
INCLUDES=/opt/rocm/miopen/include
LIB_INSTALL_PATH=/opt/rocm/miopen/lib

EXECUTABLE=./conv $(m1) $(m2) $(a1) $(a2)

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC) -I$(INCLUDES) 

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) -I$(INCLUDES) -Wl,-rpath,$(LIB_INSTALL_PATH) -L$(LIB_INSTALL_PATH) $(OBJECTS) -o $@ -lMIOpen 

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

