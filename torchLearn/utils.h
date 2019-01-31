#pragma once
#include <torch/torch.h>
#include <opencv2/highgui/highgui_c.h> //opencv input/output
#include <opencv2/imgproc/imgproc.hpp> //cvtColor

torch::Tensor imageToTensor(cv::Mat & image);

//预处理处理tensor 变量
//struct Normalize normalizeChannels({ 0.4914, 0.4822, 0.4465 }, { 0.2023, 0.1994, 0.2010 });
//tensor_image = normalizeChannels(tensor_image);
struct Normalize :torch::data::transforms::TensorTransform<>{

	Normalize(std::initializer_list<float> & means,
		std::initializer_list<float> stddevs)
		:means_(insertvalue(means)), stddevs_(insertvalue(stddevs)) {}

	std::list<torch::Tensor> insertvalue(const std::initializer_list<float> input) {

		std::list<torch::Tensor> tensorList;
		for (auto val:input)
		{
			tensorList.push_back(torch::tensor(val));
		}
		return tensorList;
	}
	torch::Tensor operator()(torch::Tensor input)
	{
		std::list<torch::Tensor>::iterator meanIter = means_.begin();
		std::list<torch::Tensor>::iterator stddevIter = stddevs_.begin();
		for (int i{ 0 }; meanIter != means_.end() && stddevIter != stddevs_.end(); ++i, ++meanIter, ++stddevIter)
		{
			input[0][i].sub_(*meanIter).div_(*stddevIter);
		}
	}
	std::list<torch::Tensor> means_, stddevs_;
};