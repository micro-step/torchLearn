#include "gpu_nms.hpp"
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition)\
/*Code block avoids redefinition of cudaError_t error */
do{\
    cudaError_t error condition;\
    if(error!=cudaSuccess){\
    std::cout<<cudaGetErrorString(error)<<std::endl;\
        \}
    }while(0)

//计算block数
#define DIVUP(m,n) ((m)/(n)+((m)%(n)>0))

//每块线程数
int const threadsPerBlock=sizeof(unsigned long long )*8;

void _set_device(int divice_id){
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if(current_device==divice_id){
        return;
    }
}


//计算IOU x1 y1 x2 y2
_device_inine float devIoU(float const * const a ,float const* const b ){
    float left =max(a[0],b[0]),right =min(a[2],b[2]);
    float top=max(a[1],b[1]),bottom=min(a[3],b[3]);
    float width =right-left;
    float height=bottom-top;
    float inters=width*height;
    float sa=(a[2]-a[0])*(a[3]-a[1]);
    float sb=(b[2]-b[0])*(b[3]-b[1]);
    return inters/(sa+sb-inters);
}

__global__ void nms_kernel(const int n_boxes,const float nms_overlap_thresh,
                            const float *dev_boxes,unsigned long long *dev_mask){
    const int row_start=blockIdx.y;
    const int col_start-blockIdx.x;

    //if (row_start> col_start) return;

    const int row_size=
            min(n_boxes-row_start*threadsPerBlock,threadsPerBlock);
    const int col_size=
            min(n_boxes-col_start*threadsPerBlock,threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock*5];

    if (threadIdx.x<col_size)




}



// keep_out(保存结果),  num_out（输出数据量）,  boxes_host 传入数据,  boxes_num 预选框数量,
// boxes_dim 预选框数据纬度 ,   nms_overlap_thresh 抑制阈值 ,  device_id 驱动设备ID
void _nms(int* keep_out,int* num_out,const float* boxes_host,int boxes_num,
    int boxes_dim,float nms_overlap_thresh,int device_id){
    _set_device(device_id);

    float* boxes_dev=NULL;

    unsigned long long* mask_dev=NULL;

    const int col_blocks=DIVUP(boxes_num,threadsPerBlock);

    CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num*boxes_dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(&boxes_dev,
                            boxes_host,
                        boxes_num*boxes_dim*sizeof(float),
                        cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&mask_dev,
                            boxes_num*col_blocks*sizeof(unsigned long long)));
    //每个线程负责计算一个预选框，预选框数量 / 每个块线程数 得到block 数量；至少一行一列
    dim3 blocks(
        DIVUP(boxes_num,threadsPerBlock),
        DIVUP(boxes_num,threadsPerBlock));
    dim3 threads(threadsPerBlock);

    nms_kernel<<blocks,threads>>(boxes_num,
                                nms_overlap_thresh,
                                boxes_dev,
                                mask_dev);
    std::vector<unsigned long long> mask_host(boxes_num*col_blocks);

    CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned,long,long)*boxes_num*col_blocks,
                        cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0],0,sizeof(unsigned long long)*col_blocks);

    int num_to_keep=0;



}
