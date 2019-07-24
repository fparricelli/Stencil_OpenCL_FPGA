#define BASE_TYPE    float
#define PARALLELISM  16

#define _MKTYPE(b,t) b ## t   
#define MKTYPE(b,t)  _MKTYPE(b,t)
#define TYPE         MKTYPE(BASE_TYPE,PARALLELISM)

#define BLOCK_DIM       16 

__constant unsigned block_dim       = BLOCK_DIM;
__constant unsigned cPar            = PARALLELISM;
__constant signed   k0              = 4; 
__constant signed   k1              = 3; 
__constant signed   k2              = 2; 
__constant signed   k3              = -3; 
__constant signed   k4              = -4; 

pipe TYPE read_to_compute     __attribute__((xcl_reqd_pipe_depth(512)));
pipe TYPE compute_to_write    __attribute__((xcl_reqd_pipe_depth(512)));

static void compute_stencil_on_block(TYPE* c_block_in,TYPE* c_block_out);
static void fill_block (__constant TYPE* restrict matrix_ptr, TYPE* c_block,const unsigned block_num, unsigned x, unsigned y );

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void read_input ( __constant TYPE* restrict vData_in,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;

    TYPE c_block[block_dim][block_dim/cPar] __attribute__((xcl_array_reshape(complete, 0)));
    BASE_TYPE* sc_block = (BASE_TYPE*) &c_block[0][0];

    for (unsigned x = 0; x < block_num; ++x){
        for (unsigned y = 0; y < block_num; ++y){
            fill_block( vData_in, &c_block[0][0], block_num, x, y );

            __attribute__((xcl_pipeline_loop))
            for(unsigned i = 0; i < block_dim; i++){
                for( unsigned j = 0; j < block_dim/cPar ; ++j){
                   write_pipe_block(read_to_compute, &c_block[i][j]);
                }
            }
        
        }
    }
}

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void compute ( size_t size,
                        size_t rows,
                        size_t cols,
                        __global TYPE* restrict sDummy )
{

    const unsigned block_num = rows / block_dim;
    TYPE c_block_in[block_dim][block_dim/cPar]  __attribute__((xcl_array_reshape(complete, 0)));
    TYPE c_block_out[block_dim][block_dim/cPar] __attribute__((xcl_array_reshape(complete, 0)));

    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            __attribute__((xcl_pipeline_loop))
            for (unsigned i = 0; i < block_dim; ++i){
            for (unsigned j = 0; j < block_dim/cPar; ++j){
                    read_pipe_block(read_to_compute, &c_block_in[i][j]);
                    }
                }
            
            compute_stencil_on_block(&c_block_in[0][0], &c_block_out[0][0]);
           
            __attribute__((xcl_pipeline_loop))
            for (unsigned i = 0; i < block_dim; ++i){
                for (unsigned j = 0; j < block_dim/cPar; ++j){
                     write_pipe_block(compute_to_write, &c_block_out[i][j]); 
                }
            }

        }
    }

}


__attribute__((reqd_work_group_size(1,1,1)))
__kernel void write_output ( __global TYPE* restrict vData_out,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;
    TYPE c_block_out[(block_dim)][(block_dim/cPar)] __attribute__((xcl_array_reshape(complete, 0)));

    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            __attribute__((xcl_pipeline_loop))
            for (unsigned i = 0; i < block_dim ; ++i){
                for (unsigned j = 0; j < block_dim/cPar; ++j){
                    read_pipe_block(compute_to_write, &c_block_out[i][j]);
                }
            }

            __attribute__((xcl_pipeline_loop))
            for (unsigned i = 0; i < block_dim ; ++i){
                for (unsigned j = 0; j < block_dim/cPar; ++j){
                    vData_out[x*block_dim*block_dim*block_num/cPar + y*block_dim/cPar + i*block_num*block_dim/cPar + j] = c_block_out[i][j];
                    }
                }
          
        }
    }
}

static void fill_block (__constant TYPE* restrict matrix_ptr, TYPE* mc_block, unsigned block_num, unsigned x, unsigned y ) {

            TYPE (*c_block)[block_dim/cPar] = (TYPE (*)[block_dim/cPar])mc_block; 

            __attribute__((xcl_pipeline_loop))
            for ( unsigned i = 0; i < block_dim; i++ ){
                for ( unsigned j = 0; j < block_dim / cPar; j++ )
                    c_block[i][j] = matrix_ptr[x*block_dim*block_dim*block_num/cPar + y*block_dim/cPar + i*block_num*block_dim/cPar + j];
            }

    
}

static void compute_stencil_on_block(TYPE* mc_block_in, TYPE* mc_block_out){


    BASE_TYPE (*c_block_in)[block_dim]   = (BASE_TYPE (*)[block_dim]) mc_block_in;  
    BASE_TYPE (*c_block_out)[block_dim]  = (BASE_TYPE (*)[block_dim]) mc_block_out;  

            __attribute__((opencl_unroll_hint))
    for ( unsigned i = 1; i < block_dim + 1; i++ ){
        for (unsigned j = 1; j < block_dim + 1; j++){
            c_block_out[i][j] = (k0*c_block_in[i][j] + k1*c_block_in[i][j+1] + k2*c_block_in[i][j-1] + k3*c_block_in[i-1][j] + k4*c_block_in[i+1][j]);

        }  
    }
}

