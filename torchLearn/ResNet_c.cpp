#include <torch//torch.h>
#include <iostream>

namespace NewRN {

	#define TN torch::nn
	TN::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planse, int64_t kerner_size,
		int64_t stride = 1, int64_t paddig = 0, bool with_bias = false) {
		TN::Conv2dOptions conv_options = TN::Conv2dOptions(in_planes, out_planse, kerner_size);
		conv_options.stride_ = stride;
		conv_options.padding_ = paddig;
		conv_options.with_bias_ = with_bias;
		return conv_options;
	}

	struct BasicBlock :TN::Module {
		static const int expansion;

		int64_t stride;
		TN::Conv2d conv1;
		TN::Conv2d conv2;
		TN::BatchNorm bn1;
		TN::BatchNorm bn2;
		TN::Sequential downsample;

		BasicBlock(int64_t inplanes,int64_t planes,int64_t stride_=1,
		TN::Sequential downsample_=TN::Sequential())
		:conv1(conv_options(inplanes,planes,3,stride_,1)),
		bn1(planes),
		conv2(conv_options(planes,planes,3,1,1)),
		bn2(planes),
		downsample(downsample)
		{
			register_module("conv1",conv1);
			register_module("bn1",bn1);
			register_module("conv2",conv2);
			register_module("bn2",bn2);
			stride=stride_;
			if (!downsample->is_empty()) {
				register_module("downsample",downsample);
			}
		}
		torch::Tensor forward(torch::Tensor x){
			torch::Tensor residual(x.clone());

			x=conv1->forward(x);
			x=bn1->forward(x);
			x=torch::relu(x);


			x=conv2->forward(x);
			x=bn2->forward(x);
			if(!downsample->is_empty())
			{
				residual=downsample->forward(residual);
			}
			x+=residual;
			x=torch::relu(x);
		}
	};
	const int BasicBlock::expansion=1;

	struct BottleNeck :TN::Module{
		static const int expansion;

		int64_t stride;

		TN::Conv2d conv1;
		TN::BatchNorm bn1;
		TN::Conv2d conv2;
		TN::BatchNorm bn2;
		TN::Conv2d conv3;
		TN::BatchNorm bn3;
		TN::Sequential downsample;

		BottleNeck(int64_t inplanes,int64_t planes,int64_t stride_=1,
		TN::Sequential downsample_=TN::Sequential())
		:conv1(conv_options(inplanes,planes,1)),
		bn1(planes),
		conv2(conv_options(planes,planes,3,stride_,1)),
		bn2(planes),
		conv3(conv_options(planes,planes*expansion,1)),
		bn3(planes),
		downsample(downsample_)
		{
			register_module("conv1",conv1);
			register_module("bn1",bn1);
			register_module("conv2",bn2);
			register_module("bn2",bn2);
			register_module("conv3",conv3);
			register_module("bn3",bn3);

			if (!downsample->is_empty()) register_module("downsample",downsample);
		}

		torch::Tensor forward(torch::Tensor x){
			torch::Tensor residual(x.clone());
			
			x=conv1->forward(x);
			x=bn1->forward(x);
			x=torch::relu(x);

			x=conv2->forward(x);
			x=bn2->forward(x);
			x=torch::relu(x);

			x=conv3->forward(x);
			x=bn2->forward(x);

			if (!downsample->is_empty()) {
				residual=downsample->forward(residual);
			}
			x+=residual;
			x=torch::relu(x);

			return x;
		}
	};

	const int BottleNeck::expansion=4;

	template<typename Block>
	struct ResNet:TN::Module{

		int64_t inplanes=64;
		TN::Conv2d conv1;
		TN::Sequential layer1;
		TN::Sequential layer2:
		TN::Sequential layer3;
		TN::Sequential layer4;
		TN::Linear fc;

		ResNet(torch::IntList layers,int64_t num_classes=1000)
		:conv1(conv_options(3,64,7,2,3)),
		bn1(64),
		layer1(_make_layer(64,layers[0])),
		layer2(_make_layer(128,layers[1],2)),
		layer3(_make_layer(256,layers[2],3)),
		layer4(_make_layer(512,layers[3],2)),
		fc(512*Block::expansion,num_classes)
		{
			register_module("conv1",conv1);
			register_module("bn1",bn1);
			register_module("layer1",layer1);
			register_module("layer2",layer2);
			register_module("layer3",layer3);
			register_module("layer4",layer4);
			register_module("fc",fc);

			for(auto m : this->modules())
			{
				if(	m.value.name()=="torch::nn::Conv2dImpl"){
					for(auto p:m.value.parameters()){
						torch::nn::init::xavier_normal_(p.value);
					}
				}
				else if (m.value.name()=="torch::nn::BatchNormImpl") {
					/* code */
					for(auto p:m.value.parameters() ){
						if (p.key=="weight") {
							torch::nn::init::constant_(p.value,1);
						}
						else if (p.key=="bias")
						{
							torch::nn::init::constant_(p.value,0);
						}
					}
				}
			}
		}
	torch::Tensor forward(torch::Tensor x){
		x=conv1->forward(x);
		x=bn1->forward(x);
		x=torch::relu(x);
		x=torch::max_pool2d(x,3,2,1);

		x=layer1->forward(x);
		x=layer2->forward(x);
		x=layer3->forward(x);
		x=layer4->forward(x);

		x=torch::avg_pool2d(x,7,1);
		x=x.view(x.sizes()[0],-1)
		x=fc->forward(x);

		return x;
	}

		private:
		TN::Sequential _make_layer(int64_t planes,int64_t blocks,int64_t stride=1){
			TN::Sequential downsample;
			if (stride!=1 or inplanes!+planes* Block::expension) {
				downsample=TN::Sequential(
					TN::Conv2d(conv_options(inplanes,planes*Block::expension,1,stride),
					TN::BatchNorm(planes*Block::expension));
			}
			TN::Sequential layers;
			layers->push_back(Block(inplanes,planes,stride,downsample));
			inplanes=planes*Block::expansion;
			for(int64_t i = 0; i < blocks; i++)
			{
				layers->push_back(Block(inplanes,planes));
				/* code */
			}
			return layers;
		}
	};

ResNet<BasicBlock> resnet18(){
	ResNet<BasicBlock> model({2,2,2,2});
	return model;
}
ResNet<BasicBlock> resnet34(){
	ResNet<BasicBlock> model({3,4,6,3});
	return model;
}

ResNet<BottleNeck> resnet50(){
	ResNet<BottleNeck> model({3,4,6,3});
	return model;
}

ResNet<BottleNeck> resnet101(){
	ResNet<BottleNeck> model({3,4,23,3});
	return model;
}
ResNet<BottleNeck> resnet152(){
	ResNet<BottleNeck> model({3,8,36,3});
	return model;
}

int main2(){
	torch::Device device("cpu");
	if (torch::cuda::is_available()) {
		/* code */
		device=torch::Device("cuda:0");
	}
	torch::Tensor t=torch::rand({2,3,244,244}).to(device);
	ResNet<BottleNeck> resnet=resnet101();
	resnet.to(device);
	t=resnet.forward(t);
	std::cout <<t.sizes()<<std::endl;
}











}
