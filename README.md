# Example of OpenCL kernel execution over Xilinx Alveo U280 Accelerator Card

## Prerequisites

To be able to compile and launch the kernels make sure that the following environment variables are properly set:
* XILINX_XRT=<path_to_xrt> (i.e /opt/xilinx/xrt/ )
* XILINX_SDX=<path_to_sdx_bin> (i.e. /opt/Xilinx/SDx/2018.3/ )
* LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${XILINX_SDX}/platforms/xilinx_u280-es1_xdma_201830_1/sw/lib/x86_64:${XILINX_XRT}/lib/:${XILINX_SDX}/lib/lnx64.o:${XILINX_SDX}/lib/lnx64.o/Default:${XILINX_SDX}/lnx64/ tools/gcc/lib64


## Compile and launch a kernel

Before building a kernel you can choose amongst three different behaviors:
* Software Emulation,
* Hardware Emulation,
* Synthesis

Software emulation does not require the synthesis and RTL implementation of the OpenCL kernel, since it simply executes as a CPU process.
Hardware emulation uses HLS tools to build the kernel, together with the memory and host interfaces. The kernel execution is emulated in the host environment.
Synthesis generates the bitstream to be flashed on the FPGA.

### Building a kernel

make build TARGET=[sw_emu | hw_emu | hw] 

### Running a kernel

make check TARGET=[sw_emu | hw_emu | hw]

The following reports are generated when the kernel execution terminates:
* Application Timeline (.wdb format) containing the timing of the execution in terms of called functions and memory accesses in a pictorial form. Data is valid only in <hw> mode.
* Profile Summary, containing text informations about the kernel execution. 
* HLS Report, containg informations about the physical resources utilization of the kernel.
 
