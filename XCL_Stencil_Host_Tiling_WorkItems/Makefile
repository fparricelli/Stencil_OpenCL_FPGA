.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make run_nimbix DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application on Nimbix Cloud."
	$(ECHO) ""
	$(ECHO) "  make aws_build DEVICE=<FPGA platform>"
	$(ECHO) "      Command to build AWS xclbin application on AWS Cloud."
	$(ECHO) ""

# Points to Utility Directory
COMMON_REPO = ../common_libs
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGETS := hw
TARGET := $(TARGETS)
DEVICE := $(DEVICES) 
 #DEVICE ?= xilinx_u280_xdma_201910_1
XCLBIN := ./xclbin

SIZE?=126

include ./utils.mk

DSA := $(call device2dsa, $(DEVICE))
BUILD_DIR := ./_x.$(TARGET).$(DSA)
KERNEL_FILE_NAME=xcl_overlap
BUILD_DIR_$(KERNEL_FILE_NAME) = $(BUILD_DIR)/$(KERNEL_FILE_NAME)

CXX := xcpp
XOCC := xocc

#Include Libraries
include $(ABS_COMMON_REPO)/libs/opencl/opencl.mk
include $(ABS_COMMON_REPO)/libs/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14
#LDFLAGS += -L$(XILINX_XRT)/lib -lxilinxopencl -lpthread #$(opencl_LDFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

HOST_SRCS += src/host.cpp

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
LDCLFLAGS += --debug --sp read_input_1.vData_in:HBM[0] --sp write_output_1.vData_out:HBM[1] --sp  compute_1.sDummy:HBM[1]

EXECUTABLE = host
CMD_ARGS = $(XCLBIN)/$(KERNEL_FILE_NAME).$(TARGET).$(DSA).xclbin

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/$(KERNEL_FILE_NAME).$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_$(KERNEL_FILE_NAME)_OBJS += $(XCLBIN)/compute.$(TARGET).$(DSA).xo
BINARY_CONTAINER_$(KERNEL_FILE_NAME)_OBJS += $(XCLBIN)/write_output.$(TARGET).$(DSA).xo
BINARY_CONTAINER_$(KERNEL_FILE_NAME)_OBJS += $(XCLBIN)/read_input.$(TARGET).$(DSA).xo

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: build
build: $(BINARY_CONTAINERS)

# Building kernel
$(XCLBIN)/read_input.$(TARGET).$(DSA).xo: src/$(KERNEL_FILE_NAME).cl
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --max_memory_ports read_input --temp_dir $(BUILD_DIR_$(KERNEL_FILE_NAME)) -c -k read_input -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/compute.$(TARGET).$(DSA).xo: src/$(KERNEL_FILE_NAME).cl
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_$(KERNEL_FILE_NAME)) -c -k compute -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/write_output.$(TARGET).$(DSA).xo: src/$(KERNEL_FILE_NAME).cl
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_$(KERNEL_FILE_NAME)) -c -k write_output -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/$(KERNEL_FILE_NAME).$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_$(KERNEL_FILE_NAME)_OBJS)
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_$(KERNEL_FILE_NAME)) -l $(LDCLFLAGS) --nk read_input:1 --nk write_output:1 --nk compute:1 -o'$@' $(+)

# Building Host
$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)
	$(CXX) -DCOLS=$(SIZE) -DROWS=$(SIZE) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	@echo "LD_FLAG: $(LDFLAGS)"
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(XCLBIN)/$(KERNEL_FILE_NAME).$(TARGET).$(DSA).xclbin
else
	 ./$(EXECUTABLE) $(XCLBIN)/$(KERNEL_FILE_NAME).$(TARGET).$(DSA).xclbin
endif
	sdx_analyze profile -i sdaccel_profile_summary.csv -f html
	sdx_analyze trace sdaccel_timeline_trace.csv

run_nimbix: all
	$(COMMON_REPO)/utility/nimbix/run_nimbix.py $(EXECUTABLE) $(CMD_ARGS) $(DSA)

aws_build: check-aws_repo $(BINARY_CONTAINERS)
	$(COMMON_REPO)/utility/aws/run_aws.py $(BINARY_CONTAINERS)

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) *profile_* *timeline_trace.* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.*

