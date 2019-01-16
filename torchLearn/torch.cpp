
// torchliblearn.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <torch/script.h>
#include<torch/torch.h>
#include<tuple>
#include <iostream>


//#define TEST_UNIT
//#define TEST_ANCHOR
//#define TEST_MODU
// Define a new Module.
struct Net : torch::nn::Module {
	Net() {
		// Construct and register two Linear submodules.
		fc1 = register_module("fc1", torch::nn::Linear(8, 64));
		fc2 = register_module("fc2", torch::nn::Linear(64, 4));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		// Use one of many tensor manipulation functions.
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, is_training());
		x = torch::sigmoid(fc2->forward(x));
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};



int main(int argc, char* argv[]) {

	torch::Tensor in = torch::randn({ 8, });

	std::cout << "in : \n" << in << std::endl;
	std::cout << in.sizes() << std::endl;

	torch::Tensor tensor = torch::randint(/*low=*/-1, /*high=*/3, { 10, 5 });
	std::cout << tensor << std::endl;
	std::cout << " tensor.options() " << std::endl;

	std::cout << tensor.options() << std::endl;
	auto index_s = tensor.slice(1, 2).squeeze();
	std::tuple<torch::Tensor,torch::Tensor> indxs = torch::max(index_s, 1);
	std::cout <<std::get<0>(indxs) << std::endl;
	std::cout << std::get<1>(indxs) << std::endl;

	std::cout << index_s << std::endl;
	//auto index_s = torch::nonzero(tensor.select(1, 2));
	//std::cout << tensor.index_select(0,index_s.squeeze()) << std::endl;

#ifdef TEST_UNIT


	std::tuple<int, int, int, int> xx = { 1, 2, 3, 4 };
	std::cout << " ( " << std::get<0>(xx) << ", " << std::get<1>(xx) << ", " << std::get<2>(xx) << " )\n";
	xx = { 4, 5, 6, 7 };

	std::cout << " ( " << std::get<0>(xx) << ", " << std::get<1>(xx) << ", " << std::get<2>(xx) << " )\n";

	int fm_sizes[] = { 8, 16, 32, 64, 128 };
	torch::Tensor tensor = torch::randint(/*low=*/-1, /*high=*/3, { 10, 5 });
	std::cout << tensor << std::endl;
	std::cout << " tensor.options() "<< std::endl;

	std::cout << tensor.options() << std::endl;
	auto index_s = tensor.slice(1, 2).squeeze();
	std::vector<torch::Tensor> indxs = torch::max(index_s, 1);
	std::cout << indxs[0] << std::endl;
	std::cout << indxs[1] << std::endl;

	std::cout << index_s << std::endl;
	//auto index_s = torch::nonzero(tensor.select(1, 2));
	//std::cout << tensor.index_select(0,index_s.squeeze()) << std::endl;
	
	torch::Tensor range_tensor = torch::range(1, 25) + c10::Scalar(0.5);
	torch::Tensor cat_tensor = torch::cat(torch::TensorList({ range_tensor, range_tensor }), 0);

	std::cout << "range_tensor : " << std::endl << range_tensor << std::endl;
	std::cout << "cat_tensor : " << std::endl << cat_tensor << std::endl;
	std::cout << ceil(9.4) << std::endl;
	torch::Tensor tin = torch::randn({ 2, 4 });
	std::cout << "Tensor tin : \n" << tin << std::endl;

	auto sigmoid_tin = tin.sigmoid();
	std::cout << "Tensor sigmoid tin : \n" << sigmoid_tin << std::endl;

	float data_in[8] = { 0.1478f,  1.4070f,  2.5251f, -1.9708f, 1.7050f, -0.1391f, -1.2296f,  0.5337f };
	torch::TensorOptions options(at::ScalarType::Float);
	std::vector<int64_t> dims = { 2, 4 };

	torch::Tensor data_tensor = torch::from_blob(data_in, at::IntList(dims), options);
	std::cout << " Data tensor : \n" << data_tensor << std::endl;
	torch::Tensor data_tensor_s = data_tensor.sigmoid();
	std::cout << " sigmoid of data tensor : \n " << data_tensor_s << std::endl;

	torch::Tensor argmax_data_tensor = data_tensor_s.argmax(1);
	std::cout << " argmax of data tensor : \n " << argmax_data_tensor << std::endl;
	dims = { 0 };
	auto em_ten = torch::empty(at::IntList(dims), options);

	std::cout << " empty tensor : \n " << em_ten.numel() << std::endl;

	auto ids = data_tensor_s > 0.5;
	std::cout << " Tensor > 0.5 : \n " << ids << std::endl;
	std::cout << " Tensor > 0.5 sizes : \n " << ids.sizes() << std::endl;
	std::cout << " Tensor > 0.5 nozero : \n " << ids.nonzero() << std::endl;

	/*coder::AnchorBox anch(600);

	dims = { static_cast<int64_t>(anch.sizeOfBoxes()), 4 };

	torch::Tensor anchor_boxes = torch::from_blob(anch.data_ptr, dims, options);

	std::cout << "Boxes [17562, 17600] : \n" << anchor_boxes.slice(0, 17562, 17600) << std::endl;

	std::cout << "Anchor Size : ( " << anch.sizeOfBoxes() << ", 4) " << std::endl;

	torch::Tensor rand_tensor = torch::randn({ 16, });
	std::cout << "Rand tensor : " << rand_tensor << std::endl;

	auto sorted_tensor = rand_tensor.sort();
	std::cout << "Sorted tensor : \n" << std::get<0>(sorted_tensor) << std::endl;
	std::cout << "Sorted Indexs : \n" << std::get<1>(sorted_tensor) << std::endl;
*/


#endif //TEST_UNIT

#ifdef TEST_ANCHOR
	coder::AnchorBox anchor(600);
	//assert(anchor.sizeOfBoxes() >= 17600);
	std::cout << anchor.sizeOfBoxes() << std::endl;
	for (int i = 17562; i < 17600; ++i) {

		std::cout << " ( " << anchor.data_ptr[4 * i] << ", " \
			<< anchor.data_ptr[4 * i + 1] << ", " \
			<< anchor.data_ptr[4 * i + 2] << ", " \
			<< anchor.data_ptr[4 * i + 3] << " ) " << std::endl;
	}
	anchor.~AnchorBox();
	std::cout << "Released " << std::endl;

#endif

#ifdef TEST_MODU
	Net model;
	std::cout << "created the network" << std::endl;
	auto out = model.forward(in);
	torch::serialize::OutputArchive outarc;
	model.save(outarc);
	outarc.save_to("xx.pt");
	std::cout << "Output: " << std::endl;
	std::cout << out << std::endl;

	torch::serialize::InputArchive inarc;
	inarc.load_from("xx.pt");

	model.load(inarc);

	//std::shared_ptr<torch::jit::script::Module> retina_module = torch::jit::load("C:/Users/Biolab/xray-retinanet/retinaxnet.pt");
	//if (retina_module != nullptr) {
	//	std::vector<torch::jit::IValue> inputs;
	//	auto x = torch::ones({ 1, 3, 600, 600 }).to(c10::kCUDA);

	//	inputs.push_back(x);
	//	retina_module->to(c10::kCUDA);
	//	for (size_t i = 0; i < 500; ++i) {
	//		auto output = retina_module->forward(inputs).toTensor();
	//		std::cout << output.sizes() << std::endl;
	//		std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/1) << '\n';
	//	}


	//	std::cout << "load retinanet successfully. " << std::endl;
	//}
	//else
	//{
	//	std::cout << "Cannot load retinanet . " << std::endl;

	//}

	std::cout << "ok\n";

#endif // TEST_MODU

	std::cout << "execute finished. " << std::endl;
	std::cin.get();
	return 0;
}


