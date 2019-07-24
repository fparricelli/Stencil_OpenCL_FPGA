#define TYPE         float 
#define V_TYPE       float2
#define V_TYPE_EXIT  float16

#define BLOCK_DIM       16 
#define PARALLELISM     2
#define OUT_PARALLELISM 16

__constant unsigned block_dim       = BLOCK_DIM;
__constant unsigned block_dim_adj   = BLOCK_DIM + 2;
__constant unsigned vDiv            = PARALLELISM;
__constant unsigned vDivOut         = OUT_PARALLELISM;
__constant signed   k0              = 4; 
__constant signed   k1              = 3; 
__constant signed   k2              = 2; 
__constant signed   k3              = -3; 
__constant signed   k4              = -4; 

pipe V_TYPE read_to_compute     __attribute__((xcl_reqd_pipe_depth(512)));
pipe V_TYPE compute_to_write    __attribute__((xcl_reqd_pipe_depth(512)));

static void compute_stencil_on_block(TYPE* c_block_in,TYPE* c_block_out);
static void fill_block (__constant TYPE* restrict matrix_ptr, TYPE* c_block,const unsigned block_num, unsigned x, unsigned y );

__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((xcl_dataflow))
__kernel void read_input ( __constant V_TYPE* restrict vData_in,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;

    V_TYPE c_block[block_dim_adj][block_dim_adj/vDiv] __attribute__((xcl_array_reshape(complete, 0)));
    TYPE* sc_block = (TYPE*) &c_block[0][0];
    __constant TYPE* restrict sData_in = (__constant TYPE* restrict) vData_in;  

    for (unsigned x = 0; x < block_num; ++x){
        for (unsigned y = 0; y < block_num; ++y){
            fill_block( sData_in, sc_block, block_num, x, y );
            
            __attribute__((xcl_pipeline_loop))
            for(unsigned i = 0; i < block_dim_adj; i++){
                __attribute__((xcl_pipeline_loop))
                for( unsigned j = 0; j < block_dim_adj/vDiv ; ++j){
                   write_pipe_block(read_to_compute, &c_block[i][j]);
                }
            }
        
        }
    }
}

__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((xcl_dataflow))
__kernel void compute ( size_t size,
                        size_t rows,
                        size_t cols,
                        __global TYPE* restrict sDummy )
{

    const unsigned block_num = rows / block_dim;
    V_TYPE c_block_in[(block_dim_adj)][(block_dim_adj/vDiv)] __attribute__((xcl_array_reshape(complete, 0)));
    V_TYPE c_block_out[(block_dim)][(block_dim/vDiv)] __attribute__((xcl_array_reshape ( complete, 0 )));

    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            for (unsigned i = 0; i < block_dim_adj; ++i){
                __attribute__((xcl_pipeline_loop))
                for (unsigned j = 0; j < block_dim_adj/vDiv; ++j){
                    read_pipe_block(read_to_compute, &c_block_in[i][j]);
                    }
                }
            
            compute_stencil_on_block((TYPE*) &c_block_in[0][0], (TYPE*) &c_block_out[0][0]);

            __attribute__((xcl_pipeline_loop))
            for (unsigned i = 0; i < block_dim; ++i){
            __attribute__((xcl_pipeline_loop))
                for (unsigned j = 0; j < block_dim/vDiv; ++j){
                     write_pipe_block(compute_to_write, &c_block_out[i][j]); 
                }
            }

        }
    }

}


__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((xcl_dataflow))
__kernel void write_output ( __global V_TYPE_EXIT* restrict vData_out,
                            size_t rows,
                            size_t cols,
                            size_t size)
{

    const unsigned block_num = rows / block_dim;
    V_TYPE_EXIT vc_block_out[(block_dim)][(block_dim/vDivOut)] __attribute__((xcl_array_reshape ( complete, 0 )));
    V_TYPE (*c_block_out)[block_dim/vDiv] = (V_TYPE (*)[block_dim/vDiv])vc_block_out;
    TYPE* sc_block_out = (TYPE*) c_block_out;

    for ( unsigned x = 0; x < block_num; ++x ){
        for (unsigned y = 0; y < block_num; ++y){
            
            for (unsigned i = 0; i < block_dim ; ++i){
            __attribute__((xcl_pipeline_loop))
                for (unsigned j = 0; j < block_dim/vDiv; ++j){
                    read_pipe_block(compute_to_write, &c_block_out[i][j]);
                }
            }

            __attribute__((opencl_unroll_hint(block_dim)))
            for (unsigned i = 0; i < block_dim ; ++i){
                __attribute__((opencl_unroll_hint(block_dim/vDivOut)))
                for (unsigned j = 0; j < block_dim/vDivOut; ++j){
                    vData_out[x*block_dim*block_dim*block_num/vDivOut + y*block_dim/vDivOut + i*block_num*block_dim/vDivOut + j] = vc_block_out[i][j];
                    }
                }
          
        }
    }
}

static void fill_block (__constant TYPE* restrict matrix_ptr, TYPE* mc_block, unsigned block_num, unsigned x, unsigned y ) {

            TYPE (*c_block)[block_dim_adj] = (TYPE (*)[block_dim_adj])mc_block; 

                __attribute__((opencl_unroll_hint(block_dim)))
            for (unsigned i = 1; i < block_dim + 1; i++ ){
                __attribute__((opencl_unroll_hint(block_dim)))
                for (unsigned j = 1; j < block_dim + 1; j++)
                    c_block[i][j] = matrix_ptr[((i-1)+x*block_dim)*block_num*block_dim + y*block_dim + j-1];
            }
            
            if ( x == 0 ){
                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for ( unsigned i = 0; i < block_dim +2 ; i++ ) {
                    c_block[0][i] = 0;
                    c_block[block_dim+1][i] = ((x+1)*block_dim*block_num*block_dim + y*block_dim + i-1) > 0 ?  matrix_ptr[(x+1)*block_dim*block_num*block_dim + y*block_dim + i-1] : 0;
                }
                
            }
            else if ( x == block_num - 1 ){
                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for (unsigned i = 0; i < block_dim + 2; i++ ){
                    c_block[0][i] = (((x-1)*block_dim)*block_num*block_dim + y*block_dim + block_num*block_dim*(block_dim - 1) + i-1) > 0 ? matrix_ptr[((x-1)*block_dim)*block_num*block_dim + y*block_dim + block_num*block_dim*(block_dim - 1) + i-1] : 0;
                    c_block[block_dim+1][i] = 0;
                }
            }
            else{
                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for ( unsigned i = 0; i < block_dim +2 ; i++ ){
                        c_block[0][i] = (((x-1)*block_dim)*block_num*block_dim + y*block_dim + block_num*block_dim*(block_dim - 1) + i-1) > 0 ? matrix_ptr[((x-1)*block_dim)*block_num*block_dim + y*block_dim + block_num*block_dim*(block_dim - 1) + i-1] : 0;
                        c_block[block_dim+1][i] = ((x+1)*block_dim*block_num*block_dim + y*block_dim + i-1) > 0 ? matrix_ptr[(x+1)*block_dim*block_num*block_dim + y*block_dim + i-1] : 0;

                }
            }

            if ( y == 0 ) {
                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for ( unsigned i = 0; i < block_dim +2 ; i++ ) {
                    c_block[i][0] = 0;
                    if( i > 0 && i < block_dim + 1 && y != block_num - 1 )
                        c_block[i][block_dim +1] = matrix_ptr[(x)*block_dim*block_num*block_dim + (y+1)*block_dim + (i-1)*block_dim*block_num];
                    else if( i > 0 && i < block_dim + 1 && y == block_num - 1 )
                        c_block[i][block_dim +1] = 0; 
                }             
            
            }
            else if (y == block_num -1 ) {

                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for ( unsigned i = 0; i < block_dim +2 ; i++ ) {
                    if( i > 0 && i < block_dim + 1 )
                        c_block[i][0] = matrix_ptr[(x)*block_dim*block_num*block_dim + (y-1)*block_dim + (i-1)*block_dim*block_num + block_dim - 1];
                    c_block[i][block_dim +1] = 0;
                }             
            }
            else {
                __attribute__((opencl_unroll_hint(block_dim + 2)))
                for ( unsigned i = 1; i < block_dim +1 ; i++ ) {
                        c_block[i][0] = matrix_ptr[(x)*block_dim*block_num*block_dim + (y-1)*block_dim + (i-1)*block_dim*block_num + block_dim - 1];
                        c_block[i][block_dim +1] = matrix_ptr[(x)*block_dim*block_num*block_dim + (y+1)*block_dim + (i-1)*block_dim*block_num];
                }   
            }    
    
}

static void compute_stencil_on_block(TYPE* mc_block_in, TYPE* mc_block_out){


    TYPE (*c_block_in)[block_dim_adj]   = (TYPE (*)[block_dim_adj]) mc_block_in;  
    TYPE (*c_block_out)[block_dim]  = (TYPE (*)[block_dim]) mc_block_out;  

    __attribute__((opencl_unroll_hint(block_dim)))
    for ( unsigned i = 0; i < block_dim; i++ ){
        __attribute__((opencl_unroll_hint(block_dim)))
        for (unsigned j = 0; j < block_dim; j++){
            c_block_out[i][j] = (k0*c_block_in[i+1][j+1] + k1*c_block_in[i+1][j+2] + k2*c_block_in[i+1][j] + k3*c_block_in[i][j+1] + k4*c_block_in[i+2][j+1]);

        }  
    }
}

