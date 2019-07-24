/*!
  * @file    host.cpp 
  * @author  CeRICT scrl - Centro Regionale di Ricerca ICT  
  * @date    Jun 2019
  * @brief   Example of a generic HBM-based OpenCL Host Application  
  *
  * This example shows how to use HBM banks to improve
  * memory bandwidth utilization.
  */

//-----------OpenCL utility layer include-----------
#include <xcl2.hpp>
//--------------------------------------------------
#include <stdlib.h>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <chrono>
#include <cstring>

#define DEBUG_LEVEL 2                   //!< Set the level of DEBUG informations to be printed: INFO = 1, INFO + DEBUG = 2, NO PRINT = 0 


//------------------------------------
//--------- Color Utility ------------
#define RED         "\033[1;31m"
#define GREEN       "\033[1;32m"
#define INFO_C      "\033[0;32m"
#define DEBUG_C     "\033[0;34m"
#define RESET       "\033[0m"
//------------------------------------
//------------------------------------

#define THRESHOLD   10000

//------------------------------------
//--------- Debug Utility ------------
#if DEBUG_LEVEL >= 1
    #define INFO(msg)    std::cout << INFO_C << "INFO: " << msg << RESET << std::endl
#else
    #define INFO(...)
#endif

#if DEBUG_LEVEL == 2
    #define DEBUG(msg)    std::cout << DEBUG_C << "DEBUG: " << msg << RESET << std::endl
#else
    #define DEBUG(...)
#endif
//------------------------------------

#define PRINT_MATRIX 1              //!< 0: No matrix is printed after the kernel execution; 1: Expected and Output matrices are printed after the kernel execution 

/**************** HBM *********************/
#define MAX_HBM_BANKCOUNT 32        //!< Maximum number of HBM banks    
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
		BANK_NAME(0), BANK_NAME(1), BANK_NAME(2), BANK_NAME(3),
		BANK_NAME(4), BANK_NAME(5), BANK_NAME(6), BANK_NAME(7),
		BANK_NAME(8), BANK_NAME(9), BANK_NAME(10), BANK_NAME(11),
		BANK_NAME(12), BANK_NAME(13), BANK_NAME(14), BANK_NAME(15),
		BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
		BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23),
		BANK_NAME(24), BANK_NAME(25), BANK_NAME(26), BANK_NAME(27),
		BANK_NAME(28), BANK_NAME(29), BANK_NAME(30), BANK_NAME(31)};
/**************** HBM *********************/

#define BLOCK_DIM  14 
#define COLS       252   
#define ROWS       252 
#define TIME_ITER 1
#define BLOCK_NUM       (COLS / BLOCK_DIM)
#define BLOCK_DIM_COLS  (BLOCK_DIM + 2)
#define BLOCK_DIM_ROWS  (BLOCK_DIM + 2)
#define ADJ_ROWS        BLOCK_DIM_ROWS * BLOCK_NUM
#define ADJ_COLS        BLOCK_DIM_COLS * BLOCK_NUM

static signed   k0              = 4; 
static signed   k1              = 3; 
static signed   k2              = 2; 
static signed   k3              = -3; 
static signed   k4              = -4; 


template <typename array_t>
array_t* adjust_input ( array_t* mat ){

    array_t* adj = new array_t[ADJ_ROWS*ADJ_COLS];

    
    for ( unsigned x = 0; x < BLOCK_NUM; ++x ){
        for ( unsigned y = 0; y < BLOCK_NUM; ++y ){
           
            for ( unsigned i = 1; i < BLOCK_DIM + 1; i++ ){
                for ( unsigned j = 1; j < BLOCK_DIM + 1; j++ ){
                    
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM+2) + j] = mat[((i-1)+x*BLOCK_DIM)*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + j-1];

                    }
            }

            if ( x == 0 ){
                for ( unsigned i = 0; i < BLOCK_DIM +2 ; i++ ) {
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i] = 0;
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + (BLOCK_DIM+1)*BLOCK_NUM*(BLOCK_DIM+2) + i] = ((x+1)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + i-1) > 0 ?  mat[(x+1)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + i-1] : 0;
                }
                
            }
            else if ( x == BLOCK_NUM - 1 ){
                for (unsigned i = 0; i < BLOCK_DIM + 2; i++ ){
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i] = (((x-1)*BLOCK_DIM)*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + BLOCK_NUM*BLOCK_DIM*(BLOCK_DIM - 1) + i-1) > 0 ? mat[((x-1)*BLOCK_DIM)*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + BLOCK_NUM*BLOCK_DIM*(BLOCK_DIM - 1) + i-1] : 0;
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + (BLOCK_DIM+1)*BLOCK_NUM*(BLOCK_DIM+2) + i] = 0;
                }
            }
            else{
                for ( unsigned i = 0; i < BLOCK_DIM +2 ; i++ ){
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i] = (((x-1)*BLOCK_DIM)*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + BLOCK_NUM*BLOCK_DIM*(BLOCK_DIM - 1) + i-1) > 0 ? mat[((x-1)*BLOCK_DIM)*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + BLOCK_NUM*BLOCK_DIM*(BLOCK_DIM - 1) + i-1] : 0;
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + (BLOCK_DIM+1)*BLOCK_NUM*(BLOCK_DIM+2) + i] = ((x+1)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + i-1) > 0 ? mat[(x+1)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + y*BLOCK_DIM + i-1] : 0;

                }
            }

            if ( y == 0 ) {
                for ( unsigned i = 0; i < BLOCK_DIM +2 ; i++ ) {
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + (BLOCK_DIM+2)*BLOCK_NUM*i] = 0;
                    if( i > 0 && i < BLOCK_DIM + 1 && y != BLOCK_NUM - 1 )
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2) +BLOCK_DIM + 1] = mat[(x)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + (y+1)*BLOCK_DIM + (i-1)*BLOCK_DIM*BLOCK_NUM];
                    else if( i > 0 && i < BLOCK_DIM + 1 && y == BLOCK_NUM - 1 )
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2) +BLOCK_DIM + 1] = 0; 
                }             
            
            }
            else if (y == BLOCK_NUM -1 ) {

                for ( unsigned i = 0; i < BLOCK_DIM +2 ; i++ ) {
                    if( i > 0 && i < BLOCK_DIM + 1 )
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2)] = mat[(x)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + (y-1)*BLOCK_DIM + (i-1)*BLOCK_DIM*BLOCK_NUM + BLOCK_DIM - 1];
                    adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2) +BLOCK_DIM + 1]= 0;
                }             
            }
            else {
                for ( unsigned i = 1; i < BLOCK_DIM +1 ; i++ ) {
                        adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2)] = mat[(x)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + (y-1)*BLOCK_DIM + (i-1)*BLOCK_DIM*BLOCK_NUM + BLOCK_DIM - 1];
                         adj[x*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + i*BLOCK_NUM*(BLOCK_DIM + 2) +BLOCK_DIM + 1] = mat[(x)*BLOCK_DIM*BLOCK_NUM*BLOCK_DIM + (y+1)*BLOCK_DIM + (i-1)*BLOCK_DIM*BLOCK_NUM];
                }   
            }    
        }
    

    }


    return adj;
}


template<typename array_t>
void print_array(array_t* input, unsigned rows, unsigned cols){

    for (unsigned i = 0; i < rows; i++){
        for (unsigned j = 0; j < cols; j++)
        std::cout << input[i*cols + j] << " ";
    std::cout << std::endl;
    }

    std::cout << std::endl;
}


/*!
 * @brief   Help function used to compute the expected result 
 * @param   input_1 is the pointer to the memory location of the first input array
 * @param   input_2 is the pointer to the memory location of the second input array
 * @param   expected is the pointer to the memory location of the output array
 * @retval  None
 */

template <typename TYPE>
void CPU_compute(TYPE* input, TYPE* expected){
 
     TYPE* temp = new TYPE[(COLS+2)*(ROWS+2)];
 
     memset(expected, 0, sizeof(TYPE)*(COLS)*(ROWS));
     memset(temp, 0, sizeof(TYPE)*(COLS+2)*(ROWS+2));
    

     for ( unsigned i = 1; i < ROWS+1; i++ )
         for ( unsigned j = 1; j < COLS+1 ; j++ )
            temp[i*(COLS+2)+j] = input[(i-1)*COLS+j-1];

     
     for(unsigned t = 0; t < TIME_ITER ; t++){
         for(signed i = 1; i < ROWS+1; i++){
             for(signed k = 1; k < COLS+1; k++){
                 expected[(i-1)*COLS + k-1] = (TYPE) (k0*temp[i*(COLS+2)+k] + k4*temp[(i+1)*(COLS+2)+k] + k3*temp[(i-1)*(COLS+2)+  k] + k2*temp[i*(COLS+2)+k-1] + k1*temp[i*(COLS+2)+k+1]);
     
             }
        }
                 for ( unsigned i = 1; i < ROWS+1; i++ )
                    for ( unsigned j = 1; j < COLS+1 ; j++ )
                        temp[(i)*(COLS+2)+j] = expected[(i-1)*(COLS)+j-1] ;
     }
     delete[] temp;
 }


/*!
 * @brief   Help function used to verify the correctness of the kernel execution 
 * @param   expected is the pointer to the memory location of the expected result array
 * @param   output is the pointer to the memory location of the output result array
 * @retval  true if arrays are equals, false otherwise 
 */
template<typename TYPE>
bool verify(TYPE* expected, TYPE* output) {

     TYPE diff;
     for(unsigned i = 0; i < COLS*ROWS ; i++){
         diff = output[i] - expected[i];
 
                 if (fabs(diff) > std::numeric_limits<TYPE>::epsilon()*THRESHOLD){
                 // if ( diff != 0 ){
                    std::cout << "Error in [" << i/COLS << " , " << i%COLS << "]" << std::endl;
                     return false;
                 }
     }
 
     return true;
 }

template<typename array_t>
array_t* adjust_output(array_t* input){

    array_t* adj = new array_t[ROWS*COLS];

    for ( unsigned x = 0; x < BLOCK_NUM ; x++ ){
        for ( unsigned y = 0; y < BLOCK_NUM ; y++ ){

            for (unsigned i = 1; i < BLOCK_DIM + 1 ; i++){
                for (unsigned j = 1; j < BLOCK_DIM + 1 ; j++){
                        adj[x*(BLOCK_DIM)*BLOCK_NUM*(BLOCK_DIM) + y*(BLOCK_DIM) + (i-1)*BLOCK_NUM*(BLOCK_DIM) + j-1] = input[(x)*(BLOCK_DIM+2)*BLOCK_NUM*(BLOCK_DIM+2) + y*(BLOCK_DIM+2) + (i)*(BLOCK_DIM+2)*BLOCK_NUM  + j];
                }

            }

        }
    }
    return adj;
}
/*!
 * @brief   Help function used to retrieve Xilinx Alveo U280 devices  
 * @param   xil_devices is the reference to a vector of devices (to be filled by the function)
 * @retval  true if the number of retrieved device is greater than zero, false otherwise
 */
bool get_devices(std::vector<cl::Device> &xil_devices) {

	xil_devices = xcl::get_xil_devices();
	cl_int err = 0;

	DEBUG("Devices retrieved: ");

	if (xil_devices.size() == 0){
		DEBUG( "No Device Found, Retry!");
        return false;
	}

	for (auto &dev : xil_devices) {
		std::string name;
		OCL_CHECK(err, dev.getInfo(CL_DEVICE_NAME, &name));
	    DEBUG(name);	
	}

    return true;

}



/*!
 * @brief   Main Function, containing the logic of kernel instantation and running.  
 * @param   XCL-BIN file of the current kernel implementation 
 * @retval  EXIT_FAILURE in case of any failure, EXIT_SUCCESS otherwise 
 */
int main ( int argc, char** argv ) {

	std::cout << "*****************************************************" << std::endl;
	std::cout << "*************** Example of XCL KERNEL  **************" << std::endl;
	std::cout << "*****************************************************" << std::endl;
	srand(time(NULL));


    //-----------Array instantation and initialization---------------
    float array_in_1[ROWS][COLS]    __attribute__((aligned(64)));
	float array_out[ADJ_ROWS][ADJ_COLS]     __attribute__((aligned(64)));
    float* adj_output;
    float* adj_input;
    //---------------------------------------------------------------
	
    for ( unsigned i = 0; i < ROWS; i++)
        for ( unsigned k = 0; k < COLS; k++)
    {
        array_in_1[i][k] = (int) (rand() % 256);
        array_out[i][k] = 0;
    }

    adj_input = adjust_input<float>(&array_in_1[0][0]);

#if PRINT_MATRIX == 1
    std::cout << "Printing input matrix\n";
    print_array<float>(&array_in_1[0][0], ROWS, COLS);

    std::cout << "Printing input matrix\n";
    print_array<float>(adj_input, ADJ_ROWS, ADJ_COLS);
#endif
    //Checking that the XCLBIN file is set 
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned fileBufSize;
    //Setting the binary file
    std::string binaryFile = argv[1];

    cl_int err;

    //Size definitions
    size_t size = ROWS*COLS;                              //Number of array elements
    size_t adj_size = ADJ_ROWS*ADJ_COLS;                              //Number of array elements
    size_t rows = ROWS;                              //Number of array elements
    size_t cols = COLS;                              //Number of array elements
    size_t adj_rows = ADJ_ROWS;                              //Number of array elements
    size_t adj_cols = ADJ_COLS;                              //Number of array elements
    size_t time_iter = TIME_ITER;                              //Number of array elements
    size_t input_size_bytes = sizeof(float) * adj_size;  //Byte-size of input-arrays
    size_t output_size_bytes = sizeof(float) * adj_size;        //Byte-size of output-arrays

    //Getting devices
    std::vector<cl::Device> devices;
    if ( !get_devices(devices) )
        return EXIT_FAILURE;


    //-----------OCL Runtime initialization--------------------------
    OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q0(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |  CL_QUEUE_PROFILING_ENABLE, &err));
    //---------------------------------------------------------------

    //-----------OCL Program and Kernel initialization-----------------
	char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
	cl::Program::Binaries bins{{fileBuf, fileBufSize}};

	OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

	OCL_CHECK(err, cl::Kernel rd_kernel(program, "read_input", &err));
	OCL_CHECK(err, cl::Kernel cu_kernel(program, "compute", &err));
	OCL_CHECK(err, cl::Kernel wr_kernel(program, "write_output", &err));
    //------------------------------------------------------------------

    //-----------OCL Buffer initialization-----------------------------
	cl_mem_ext_ptr_t inBufExt_1, outBufExt;

    INFO("[HOST] - Buffer Allocation: ");
    INFO("[INPUT #1] -> bank 0 ");
    INFO("[ OUTPUT ] -> bank 1 ");

    inBufExt_1.obj = adj_input;
    inBufExt_1.param = 0;
    inBufExt_1.flags = bank[0];

    outBufExt.obj = array_out;
    outBufExt.param = 0;
    outBufExt.flags = bank[1];

    OCL_CHECK(err, cl::Buffer buffer_in1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
           input_size_bytes, &inBufExt_1, &err));
    
    OCL_CHECK(err, cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
            output_size_bytes, &outBufExt, &err));
    //------------------------------------------------------------------
    
    //----------Set the kernel arguments-------------------------
    OCL_CHECK(err, err = rd_kernel.setArg(0, buffer_in1));
    OCL_CHECK(err, err = rd_kernel.setArg(1, adj_rows));
    OCL_CHECK(err, err = rd_kernel.setArg(2, adj_cols));
    OCL_CHECK(err, err = rd_kernel.setArg(3, input_size_bytes));

    OCL_CHECK(err, err = cu_kernel.setArg(0, input_size_bytes));
    OCL_CHECK(err, err = cu_kernel.setArg(1, adj_rows));
    OCL_CHECK(err, err = cu_kernel.setArg(2, adj_cols));
    OCL_CHECK(err, err = cu_kernel.setArg(3, buffer_output));
    
    OCL_CHECK(err, err = wr_kernel.setArg(0, buffer_output));
    OCL_CHECK(err, err = wr_kernel.setArg(1, adj_rows));
    OCL_CHECK(err, err = wr_kernel.setArg(2, adj_cols));
    OCL_CHECK(err, err = wr_kernel.setArg(3, output_size_bytes));
    //------------------------------------------------------------------
   
    //Writing on global memory...
    OCL_CHECK(err, err = q0.enqueueMigrateMemObjects({buffer_in1},0/* 0 means from host*/));
    
    cl::Event event;
    
    auto kernel_start = std::chrono::high_resolution_clock::now();

    //---------------Kernels Launch--------------------
    DEBUG("Running <read_input>..."); 
    OCL_CHECK(err, err = q0.enqueueNDRangeKernel(rd_kernel, 0, 1, 1));//, 0, DATA_SIZE, DATA_SIZE));
    DEBUG("Running <compute>...");
    OCL_CHECK(err, err = q0.enqueueNDRangeKernel(cu_kernel, 0, 1, 1));//, 0, DATA_SIZE, DATA_SIZE));
    DEBUG("Running <write_output>...");
    OCL_CHECK(err, err = q0.enqueueNDRangeKernel(wr_kernel, 0, 1, 1));//, 0, DATA_SIZE, DATA_SIZE));
    INFO("Waiting kernel ending...");
    OCL_CHECK(err, err = q0.finish());
    //------------------------------------------------------------------
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
   
    //Reading back computation results
    DEBUG("Reading Data from device...");
    OCL_CHECK(err, err = q0.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
    
    DEBUG("Waiting for pending operations...");
    OCL_CHECK(err, err = q0.finish());

    auto kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

    INFO("Kernel Time: " << kernel_time.count()); 

    delete[] fileBuf;

    adj_output = adjust_output<float>(&array_out[0][0]);
    float expected[ROWS*COLS];

    //---------------Kernel correctness verification--------------------
    CPU_compute<float>(&array_in_1[0][0], &expected[0]);

    if(verify<float>(&expected[0], adj_output)) std::cout << GREEN << "Test Passed";  else std::cout << RED << "Test NOT Passed";
    std::cout << RESET << std::endl;
#if PRINT_MATRIX == 1
    std::cout << "Printing expected array...\n";
    print_array<float>(&expected[0], ROWS, COLS);
    std::cout << "Printing obtained array...\n";
    print_array<float>(adj_output, ROWS, COLS);
#endif
    //------------------------------------------------------------------
	return EXIT_SUCCESS;
}






