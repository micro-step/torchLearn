#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Darknet.h"

using namespace torch;

int mainOK(int argc, const char* argv[])
{
	/*unsigned long long t = 0;
	for (int i = 0; i < 5; i++) {
			t |= 1ULL << i;//从右到左 先移位 再取或| 然后赋值；
			std::cout << t << endl;
	}*/

	if (argc!=2)
	{
		std::cout << "usage: yolo-app <image path>\n";
		//return -1;
	}
	cout << _pgmptr << endl;
	torch::DeviceType device_type;

	if (torch::cuda::is_available())
	{
		device_type = torch::kCUDA;
	}
	else
	{
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	//input image size for yolo v3
	int input_image_size = 416;

	Darknet net("E:\\PyL\\torchLearn\\x64\\Debug\\models\\yolov3.cfg", &device);

	map<string, string> *info = net.get_net_info();
	info->operator []("height") = std::to_string(input_image_size);

	std::cout << "loading weight ..." << endl;
	net.load_weights("E:\\PyL\\torchLearn\\x64\\Debug\\models\\yolov3.weights");
	std::cout << "weight loaded...." << endl;

	net.to(device);

	torch::NoGradGuard no_grad;

	net.eval();

	std::cout << "inference ..." << endl;

	cv::Mat origin_image, resized_image;

	origin_image = cv::imread("E:\\PyL\\torchLearn\\x64\\Debug\\models\\dog.jpg");
	//origin_image = cv::imread(argv[1]);

	cv::cvtColor(origin_image, resized_image, CV_BGR2RGB);
	cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

	cv::Mat img_float;
	resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

	auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, { 1, input_image_size, input_image_size, 3 });
	img_tensor = img_tensor.permute({ 0,3, 1, 2 });
	//std::cout << img_tensor.sizes() << std::endl;

	//std::cout << "img_var.sizes" << img_tensor.sizes() << endl;

	at::Tensor img_var = torch::autograd::make_variable(img_tensor).to(device);

	//std::cout << "img_var.sizes" << img_var.sizes()<<endl;

	at::Tensor output = net.forward(img_var);

//	std::cout << output.slice(1, 1, 11) << std::endl;

	//fillter result by NMS
	//class_num =80
	//confidence =0.6

	auto result = net.write_results(output, 80, 0.7, 0.1);
	//std::cout << result << std::endl;

	if (result.dim()==1)
	{
		std::cout << "no object found" << endl;
	}
	else
	{
		int obj_num = result.size(0);
		std::cout << obj_num << "object found" << endl;
		float w_scale = float(origin_image.cols) / input_image_size;
		float h_scale = float(origin_image.rows) / input_image_size;

		auto result_data = result.accessor<float,2>();
		std::cout << result.sizes() << std::endl;
		for (int i = 0; i < result.size(0);i++)
		{
			cv::rectangle(origin_image, cv::Point(result_data[i][1]* w_scale, result_data[i][2]* h_scale), cv::Point(result_data[i][3]* w_scale, result_data[i][4]* h_scale), cv::Scalar(0, 0, 255), 1, 1, 0);
		}
		//std::cout << result << std::endl;
		cv::imwrite("e:\out-det.jpg", origin_image);
	}
	std::cout << "Done" << endl;
	return 0;
}