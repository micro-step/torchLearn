#ifndef CAFFE2NET_H
#define CADDE2NET_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opecv.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

using namespace std;
using namespace cv;
using namespace caffe2;

class Caffe2Net{
    public:
    Caffe2Net(string initNet,string predictNet);
    virtual ~Caffe2Net()=0;
    vector<flaot> predict(Mat img);

    protected:
    virtual TensorCPU preProcess(Mat img)=0;
    virtual vector<flaot> postProcess(TensorCPU,output)=0;

    Workspace worksapce ;
    unique_ptr<NetBase> predict_net;
}
#endif