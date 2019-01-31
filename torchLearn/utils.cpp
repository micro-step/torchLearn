#include "utils.h"

torch::Tensor imageToTensor(cv::Mat & image) {
	
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	cv::Mat rgb[3];

	cv::split(image, rgb);

	cv::Mat rgbConcat;
	cv::vconcat(rgb, 3, rgbConcat);

	torch::Tensor tensor_image = torch::from_blob(rgbConcat.data,
		{ 1,image.channels(),image.rows,image.cols}, at::kByte);

	tensor_image = tensor_image.toType(at::kFloat);
	tensor_image = tensor_image.div(255);
	return tensor_image;
}
