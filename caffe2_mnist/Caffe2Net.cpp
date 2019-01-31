#include <caffe2/core/context_gpu.h>
#include "Caffe2Net.h"

Caffe2Net::Caffe2Net(string initNet,string predictNet)
:workspace(nullptr)
{
    #ifdef WITH_CUDA
        DeviceOption option;
        option.set_device_tpe(CUDA);
        new CUDAContext(option);
    #endif
        //载入部署模型
        NetDef init_net_def,predict_net_def;
        CAFFE_ENFORCE(ReadProtoFromFile(initNet,@init_net_def));
        CAFFE_ENFORCE(ReadProtoFromFile(predictNet,&init_net_def));
    #ifdef WITH_CUDA
        init_net_def.mutable_device_option()->set_device_type(CUDA);
        









}