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

static void compute_stencil_on_block(local TYPE* c_block_in,local TYPE* c_block_out, const int idx);
static void fill_block (__constant TYPE* restrict matrix_ptr, local TYPE* c_block,const unsigned block_num, unsigned x, unsigned y, const int idx );

__attribute__((reqd_work_group_size(16,1,1)))
__kernel void read_input ( __constant TYPE* restrict vData_in,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;

    const int idx = get_global_id(0);

    local TYPE c_block[block_dim][block_dim/cPar] __attribute__((xcl_array_reshape(complete, 0)));
    local BASE_TYPE* sc_block = (local BASE_TYPE*) &c_block[0][0];

    for (unsigned x = 0; x < block_num; ++x){
        for (unsigned y = 0; y < block_num; ++y){
            fill_block( vData_in, &c_block[0][0], block_num, x, y, idx );

            barrier(CLK_LOCAL_MEM_FENCE);

            __attribute__((xcl_pipeline_loop))
            for( unsigned j = 0; j < block_dim/cPar ; ++j)
                   write_pipe_block(read_to_compute, &c_block[idx][j]);
                
            
        
        }
    }
}

__attribute__((reqd_work_group_size(16,1,1)))
__kernel void compute ( size_t size,
                        size_t rows,
                        size_t cols,
                        __global TYPE* restrict sDummy )
{

    const unsigned block_num = rows / block_dim;
    local TYPE c_block_in[block_dim][block_dim/cPar]  __attribute__((xcl_array_reshape(complete, 0)));
    local TYPE c_block_out[block_dim][block_dim/cPar] __attribute__((xcl_array_reshape(complete, 0)));

    const int idx = get_global_id(0);

    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            __attribute__((xcl_pipeline_loop))
            for (unsigned j = 0; j < block_dim/cPar; ++j)
                 read_pipe_block(read_to_compute, &c_block_in[idx][j]);
    
            barrier (CLK_LOCAL_MEM_FENCE);
            
            compute_stencil_on_block(&c_block_in[0][0], &c_block_out[0][0], idx);
            
            __attribute__((xcl_pipeline_loop))
            for (unsigned j = 0; j < block_dim/cPar; ++j)
                 write_pipe_block(compute_to_write, &c_block_out[idx][j]); 

        }
    }

}


__attribute__((reqd_work_group_size(16,1,1)))
__kernel void write_output ( __global TYPE* restrict vData_out,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;
    local TYPE c_block_out[(block_dim)][(block_dim/cPar)] __attribute__((xcl_array_reshape(complete, 0)));
    const int idx = get_global_id(0);


    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            __attribute__((xcl_pipeline_loop))
                for (unsigned j = 0; j < block_dim/cPar; ++j)
                    read_pipe_block(compute_to_write, &c_block_out[idx][j]);

                    barrier(CLK_LOCAL_MEM_FENCE);

            __attribute__((opencl_unroll_hint))
                for (unsigned j = 0; j < block_dim/cPar; ++j)
                    vData_out[x*block_dim*block_dim*block_num/cPar + y*block_dim/cPar + idx*block_num*block_dim/cPar + j] = c_block_out[idx][j];
                    
                
          
        }
    }
}

static void fill_block (__constant TYPE* restrict matrix_ptr, local TYPE* mc_block, unsigned block_num, unsigned x, unsigned y, const int idx ) {

            local TYPE (*c_block)[block_dim/cPar] = (local TYPE (*)[block_dim/cPar])mc_block; 

            __attribute__((xcl_pipeline_loop))
            for ( unsigned j = 0; j < block_dim / cPar; j++ )
                    c_block[idx][j] = matrix_ptr[x*block_dim*block_dim*block_num/cPar + y*block_dim/cPar + idx*block_num*block_dim/cPar + j];
            

    
}

static void compute_stencil_on_block(local TYPE* mc_block_in, local TYPE* mc_block_out, const int idx){


    local BASE_TYPE (*c_block_in)[block_dim]   = (local BASE_TYPE (*)[block_dim]) mc_block_in;  
    local BASE_TYPE (*c_block_out)[block_dim]  = (local BASE_TYPE (*)[block_dim]) mc_block_out;  
        
        int i = idx + 1; 

        __attribute__((xcl_unroll_hint))
        for (unsigned j = 1; j < block_dim +1 ; j++){
            c_block_out[i][j] = (k0*c_block_in[i][j] + k1*c_block_in[i][j+1] + k2*c_block_in[i][j-1] + k3*c_block_in[i -1][j] + k4*c_block_in[i+1][j]);

        }
}



