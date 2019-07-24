# Example of OpenCL kernel execution over Xilinx Alveo U280 Accelerator Card

This example should be considered as a pattern for building more complex kernel.
It is optimized in terms of memory transfer between global and local memory, maximizing the data bandwidth using HBM banks. The kernel itself simply does a vector addition between two input
data arrays with a size that must be multiple of <TYPE> parallelism. By default, <TYPE> is set as "uint16". 

## Host-Application Hints
Host application is based on the <xcl2.hpp> library, that provides an abstraction view of the OpenCL APIs in an OO manner. The application control flow, however, follows the standard way to build an OpenCL host runnable. Since we need to store buffers on HBM banks, <cl_mem_ext_ptr> together CL_MEM_EXT_PTR_XILINX  must be used to tell the compiler the physical location on which buffers should be read/written. cl_mem_ext_ptr is a struct containing a field named flags in which
you put <n_bank> | XCL_MEM_TOPOLOGY to set the bank in which you want to access.
CL_MEM_EXT_PTR_XILINX must be used as a cl_mem_flags while creating the buffer in clCreateBuffer(...).

## Kernel-Function Hints
The acceleration function is structured into three kernels, running in a pipelined and overlapped manner. This function separation realizes the isolation between computation logic and memory access logic. As a result the following ones are considered:
* READ INPUT, accesses global memory (RD) storing data in size-bounded local buffers. Data is then sent to the next step through pipes. The number of pipe used is equal to the number of global memory banks accessed. 
* COMPUTE, contains the core logic of the kernel. It gets data from the <READ_INPUT> pipes and put the results into an output pipe, directed to the next step.
* WRITE_OUTPUT, accesses global memory (WR) to make computation results accessible by host. 

Kernels communicates through OpenCL pipes of a fixed size as the following schema shows:

```
 ______________              ______________             ______________
|              |            |              |           |              |
|  READ_INPUT  |==========> |   COMPUTE    |---------> | WRITE_OUTPUT |
|______________|  pipes[]   |______________|   pipe    |______________|
```
The example kernel infers a 512-bit AXI4 interface between the OpenCL kernel and the global memory by setting TYPE as uint16. Each global pointer is passed with the attribute <restrict> to allow compiler optimizations on pointer usage. 

To improve the lenght of the burst transactions, a buffer layer is placed between the memory access and the pipe push/pop operation. The buffer is handled in a circular manner and its size can be statically defined. 

Task-level pipelining within the kernel is improved through the usage of the attribute xcl_dataflow.

## Makefile
Before running synthesis and implementation, you should tell XOCC about the kernels you want to compile, the clock frequency, the number of compute units to be instantiated and other factors. 

The KERNEL_FILE_NAME variable contains the name of the OpenCL file in which kernel functions reside. 

LDCLFLAGS variable contains linking-related arguments. Here you have to specify the interfaces among kernel memory interfaces and the global memory by using the --sp tag as below:
    --sp <kernel_instance>.<argument>:HBM[<bank>]

The BINARY_CONTAINERS variable holds the xclbin file name.
The BINARY_CONTAINER_$(KERNEL_FILE_NAME)_OBJS variable holds the name the builded kernels (in .xo format).

While linking the .xclbin you're able to instantiate the number of kernel instances through the --nk tag. For instance:
    --nk <kernel_name>:<number_of_instances> 
By default, each kernel function is instantiated only once.

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
 
