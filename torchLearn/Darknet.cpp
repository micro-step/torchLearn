#include "Darknet.h"
#include <stdio.h>
#include <typeinfo>
#include <iostream>
#include <ctype.h>
//trim from start (in place)
static inline void ltrim(std::string &s){
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch){
		return !isspace(ch);
	}));
}
//trim from end (in place)
static inline void	rtrim(std::string &s){
	s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch){
		return !isspace(ch);
	}).base(),s.end());
}

//trim from both ends (in place)
static inline void trim(std::string &s){
	ltrim(s);
	rtrim(s);
}

static inline int split(const string& str, std::vector<string>& ret_, string sep = ",")
{
	if (str.empty())
	{
		return 0;
	}
	string tmp;
	string::size_type pos_begin = str.find_first_not_of(sep);
	string::size_type comma_pos = 0;
	while (pos_begin!=string::npos)
	{
		comma_pos = str.find(sep, pos_begin);
		if (comma_pos!=string::npos)
		{
			tmp = str.substr(pos_begin, comma_pos - pos_begin);
			pos_begin = comma_pos + sep.length();
		}
		else
		{
			tmp = str.substr(pos_begin);
			pos_begin = comma_pos;
		}
		if (!tmp.empty())
		{
			trim(tmp);
			ret_.push_back(tmp);
			tmp.clear();
		}
	}
	return 0;
}

static inline int split(const string& str, std::vector<int>& ret_, string sep = ","){
	std::vector<string> tmp;
	split(str, tmp, sep);
	for (int i = 0; i < tmp.size();i++)
	{
		ret_.push_back(std::stoi(tmp[i]));
	}
	return 0;
}

static inline torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2){
	//Get the coordinates of bounding boxes
	torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
	b1_x1 = box1.select(1, 0);
	b1_y1 = box1.select(1, 1);
	b1_x2 = box1.select(1, 2);
	b1_y2 = box1.select(1, 3);

	torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
	b2_x1 = box2.select(1, 0);
	b2_y1 = box2.select(1, 1);
	b2_x2 = box2.select(1, 2);
	b2_y2 = box2.select(1, 3);

	torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);
	torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);
	torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);
	torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);

	//Intersection area
	torch::Tensor inter_area = torch::min(inter_rect_x2 - inter_rect_x1, torch::zeros(inter_rect_x1.sizes()))*
		torch::min(inter_rect_y2 - inter_rect_y1, torch::zeros(inter_rect_y2.sizes()));

	torch::Tensor b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1);
	torch::Tensor b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1);
	torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);
	return iou;

}
int Darknet::get_int_from_cfg(map<string, string> block, string key, int default_value){
	if (block.find(key)!=block.end())
	{
		return std::stoi(block.at(key));
	}
	return default_value;
}
string Darknet::get_string_from_cfg(map<string, string> block, string key, string default_value)
{
	if (block.find(key)!=block.end())
	{
		return block.at(key);
	}
	return default_value;
}

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
	int64_t stride /* = 1 */, int64_t padding /* = 0 */, int64_t groups, bool with_bias = false /* = false */){
	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
	conv_options.stride_ = stride;
	conv_options.padding_ = padding;
	conv_options.groups_ = groups;
	conv_options.with_bias_ = with_bias;

	return conv_options;
}

torch::nn::BatchNormOptions bn_options(int64_t features){
	torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
	bn_options.affine_ = true;
	bn_options.stateful_ = true;
	return bn_options;
}

struct EmptyLayer :torch::nn::Module
{
	EmptyLayer(){}
	torch::Tensor forward(torch::Tensor x){ return x; }
};

struct UpsampleLayer:torch::nn::Module
{
	int _stride;
	UpsampleLayer(int stride){
		_stride = stride;
	}

	torch::Tensor forward(torch::Tensor x){
		torch::IntList sizes = x.sizes();
		int64_t w, h;
		if (sizes.size()==4)
		{
			w = sizes[2] * _stride;
			h = sizes[3] * _stride;
			x = torch::upsample_nearest1d(x, { w, h });
		}
		else if (sizes.size()==3)
		{
			w = sizes[2] * _stride;
			x = torch::upsample_nearest1d(x, { w });
		}
		return x;
	}
};

struct DetectionLayer:torch::nn::Module
{
	vector<float> _anchors;
	DetectionLayer(vector<float> anchors){
		_anchors = anchors;
	}

	torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device){
		return predict_transform(prediction, inp_dim, _anchors, num_classes, device);
	}
	//inp_dim 输入图片维度 宽/高
	torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, vector<float> anchors,
		int num_classes, torch::Device device) {

		int batch_size = prediction.size(0);//批次数量
		int stride = floor(inp_dim/prediction.size(2));//缩放比例
		int grid_size = floor(inp_dim / stride);//网格每行数量
		int bbox_attrs = 5 + num_classes;//每行数据个数 中心坐标偏移量 + 宽高偏移量+ 目标分数=5 + 种类分数；
		int num_anchors = anchors.size() / 2;// anchors 每两个（宽高）为一组

		for (int i=0;i<anchors.size();++i)
		{
			anchors[i] = anchors[i] / stride;//缩anchor 宽高
		}
		//将输出向量转换为 batch_size x bbox_attrs*num_anchor x grid_size*grid_size 维度的向量
		torch::Tensor result = prediction.view({ batch_size,bbox_attrs*num_anchors,grid_size*grid_size });
		result = result.transpose(1, 2).contiguous();//矩阵转置
		result = result.view({ batch_size, grid_size*grid_size*num_anchors, bbox_attrs });//矩阵变换

		/*
		正常情况下，YOLO 不会预测边界框中心的确切坐标。它预测：
		● 与预测目标的网格单元左上角相关的偏移；
		● 使用特征图单元的维度（1）进行归一化的偏移。
		以我们的图像为例。如果中心的预测是 (0.4, 0.7)，则中心在 13 x 13 特征图上的坐标是 (6.4, 6.7)（红色单元的左上角坐标是 (6,6),(6,6)）+(0.4, 0.7)）。
		但是，如果预测到的 x,y 坐标大于 1，比如 (1.2, 0.7)。那么中心坐标是 (7.2, 6.7)。注意该中心在红色单元右侧的单元中，或第 7 行的第 8 个单元。这打破了 YOLO 背后的理论，因为如果我们假设红色框负责预测目标狗，那么狗的中心必须在红色单元中，不应该在它旁边的网格单元中。
		因此，为了解决这个问题，我们对输出执行 sigmoid 函数，将输出压缩到区间 0 到 1 之间，有效确保中心处于执行预测的网格单元中。
		
		Object 分数表示目标在边界框内的概率。红色网格和相邻网格的 Object 分数应该接近 1，而角落处的网格的 Object 分数可能接近 0。
		objectness 分数的计算也使用 sigmoid 函数，因此它可以被理解为概率。
		*/
		result.select(2, 0).sigmoid_();//中心 tx 偏移量
		result.select(2, 1).sigmoid_();//中心 ty 偏移量
		result.select(2,4).sigmoid_();//目标分数 


		auto grid_len = torch::arange(grid_size);

		vector<torch::Tensor> args = torch::meshgrid({ grid_len,grid_len });
		torch::Tensor x_offset = args[1].contiguous().view({ -1, 1 });//变成一列向量
		torch::Tensor y_offset = args[0].contiguous().view({ -1, 1 });

		cout << "x_offset" << x_offset << endl;
		cout << "y_offset" << y_offset << endl;

		x_offset = x_offset.to(device);
		y_offset = y_offset.to(device);

		//按列拼接，形成A=(grid_len * grid_len* grid_len * grid_len）行 一列的矩阵 
		//然后复制num_anchors 个 变为  A行 * num_anchors列 的矩阵 然后变形为 x行 2列的矩阵 x=1/2 * A * num_anchors 

		auto x_y_offset = torch::cat({ x_offset,y_offset },1).repeat({ 1,num_anchors }).view({ -1,2 }).unsqueeze(0);

		result.slice(2, 0, 2).add_(x_y_offset);//slice 选择第2+1 纬 的0 1 列数据即 中心的偏移量
		torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), { num_anchors, 2 });

		result.slice(2, 2, 4).exp_().mul_(anchors_tensor); //选择anchor的宽高

		/*类别置信度
		类别置信度表示检测到的对象属于某个类别的概率（如狗、猫、香蕉、汽车等）。在 v3 之前，YOLO
		需要对类别分数执行 softmax 函数操作。
		但是，YOLO v3 舍弃了这种设计，作者选择使用 sigmoid 函数。因为对类别分数执行 softmax
		操作的前提是类别是互斥的。简言之，如果对象属于一个类别，那么必须确保其不属于另一个类别。
		这在我们设置检测器的 COCO 数据集上是正确的。但是，当出现类别「女性」（Women）和「人」（Person）时，
		该假设不可行。这就是作者选择不使用 Softmax 激活函数的原因。*/
		result.slice(2, 5, 5 + num_classes).sigmoid_();//  5：5+num_classes 
		
		 //坐标乘以 缩放量得到 真实的坐标和宽高 （1：2 的所有行的0：4 的数据乘以 stride）
		result.slice(2, 0, 4).mul_(stride);

		return result;
	}
};

//Darknet
Darknet::Darknet(const char *conf_file, torch::Device *device){
	load_cfg(conf_file);
	_device = device;
	create_modules();
}

void Darknet::load_cfg(const char *cfg_file){
	fstream fs(cfg_file);
	if (!fs)
	{
		std::cout << " Fail to load cfg file:" << cfg_file << std::endl;
		return;
	}
	string line;
	while (getline(fs,line))
	{
		trim(line);
		if (line.empty())
		{
			continue;
		}
		if (line.substr(0,1)=="[")
		{
			map<string, string> block;
			string key = line.substr(1, line.length() - 2);
			block["type"] = key;
			blocks.push_back(block);
		}
		else if (line.substr(0, 1) == "#")
			continue;
		else
		{
			map<string, string> *block = &blocks[blocks.size() - 1];
			vector<string> op_info;
			split(line, op_info, "=");
			if (op_info.size()==2)
			{
				string p_key = op_info[0];
				string p_value = op_info[1];
				block->operator [](p_key) = p_value;
			}
		}
	}
	fs.close();
}

void Darknet::create_modules(){
	int prev_filters = 3;//输入channel
	std::vector<int> output_filters;
	int index = 0;
	int filters = 0;//conv channel 该层输出channel;
	for (int i = 0, len = blocks.size(); i < len;i++)
	{
		map<string, string> block=blocks[i];
		torch::nn::Sequential module;
		string layer_type = block["type"];
		if (layer_type == "net")
			continue;
		if (layer_type=="convolutional")
		{
			string activation = get_string_from_cfg(block, "activation", "");
			int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
			filters = get_int_from_cfg(block, "filters", 0);
			int padding = get_int_from_cfg(block, "pad", 0);
			int stride = get_int_from_cfg(block, "stride", 0);
			int kernel_size = get_int_from_cfg(block, "size", 0);

			int pad = padding > 0 ? (kernel_size - 1) / 2 : 0;
			bool with_bias = batch_normalize > 0 ? true : false;
			torch::nn::Conv2d conv =
				torch::nn::Conv2d(conv_options(prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
			module->push_back(conv);
			if (batch_normalize>0)
			{
				torch::nn::BatchNorm bn = torch::nn::BatchNorm(bn_options(filters));
				module->push_back(bn);
			}
			if (activation=="leaky")
			{
				module->push_back(torch::nn::Functional(torch::leaky_relu,/*slope=*/0.1));
			}
		}
		else if (layer_type=="upsample")
		{
			int stride = get_int_from_cfg(block, "stride", 1);
			UpsampleLayer uplayer(stride);
			module->push_back(uplayer);
		}
		else if (layer_type=="shortcut")
		{
			//skip connnection
			int from = get_int_from_cfg(block, "from", 1);
			block["from"] = std::to_string(from);
			blocks[i] = block;

			//placeholder
			EmptyLayer layer;
			module->push_back(layer);
		}
		else if (layer_type=="route")
		{
			//L 85: -1,61
			string layers_info = get_string_from_cfg(block, "layers", "");

			std::vector<string> layers;
			split(layers_info, layers, ",");

			std::string::size_type sz;
			signed int start = std::stoi(layers[0]);
			signed int end = 0;
			if (layers.size() >1)
			{
				end = std::stoi(layers[1]);
			}
			if (end>0)end = end - index;
			if (start > 0) start = start - index;
			
			block["start"] = start;
			block["end"] = end;

			blocks[i] = block;

			//placeholder
			EmptyLayer layer;
			module->push_back(layer);
			if (end<0)
			{
				filters = output_filters[index + start] + output_filters[index + end];
			}
			else
			{
				filters = output_filters[index + start];
			}
		}
		else if (layer_type=="yolo")
		{
			string mask_info = get_string_from_cfg(block, "mask", "");
			std::vector<int> masks;
			split(mask_info, masks, ",");

			string anchor_info = get_string_from_cfg(block, "anchors", "");
			vector<int> anchors;
			split(anchor_info, anchors, ",");

			std::vector<float> anchor_points;
			int pos;
			for (int i=0;i<masks.size();++i)
			{
				pos = masks[i];
				anchor_points.push_back(anchors[pos * 2]);
				anchor_points.push_back(anchors[pos * 2 + 1]);
			}
			DetectionLayer layer(anchor_points);
			module->push_back(layer);
		}
		else
		{
			cout << "unsupported operator:" << layer_type << endl;
		}

		prev_filters = filters;
		output_filters.push_back(filters);
		module_list.push_back(module);
		char *module_key = new char[strlen("layer_") + sizeof(index) + 1];
		sprintf(module_key, "%s%d", "layer_", index);
		register_module(module_key, module);
		index += 1;
	}
}

map<string, string> * Darknet::get_net_info() {
	if (blocks.size()>0)
	{
		return &blocks[0];
	}
}
void Darknet::load_weights(const char *weight_file){
	
	ifstream fs(weight_file,ios::binary);

	//header info :5*int32_t
	int32_t header_size = sizeof(int32_t)*5;

	int64_t index_weight = 0;
	
	fs.seekg(0, fs.end);

	int64_t lenght = fs.tellg();

	//skip header
	lenght = lenght - header_size;
	fs.seekg(header_size, fs.beg);

	float*weight_src = (float*)malloc(lenght);

	fs.read(reinterpret_cast<char*>(weight_src), lenght);

	fs.close();
	at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).is_variable(true);
	at::Tensor weight = torch::CPU(torch::kFloat32).tensorFromBlob(weight_src, { lenght / 4 });


	for (int i = 0; i < module_list.size();i++)
	{
		map<string, string> module_info = blocks[i + 1];

		string module_type = module_info["type"];

		//only conv layer need to load weight

		if (module_type!="convolutional") continue;

		torch::nn::Sequential seq_module = module_list[i];

		auto conv_module = seq_module.ptr()->ptr(0);
		torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl*>(conv_module.get());

		int batch_normlize = get_int_from_cfg(module_info, "batch_normalize", 0);

		if (batch_normlize>0)
		{
			//second module
			auto bn_module = seq_module.ptr()->ptr(1);

			torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

			int num_bn_biases = bn_imp->bias.numel();
			// s=（x-m）除以(方差+s）x=w*s +b;
			at::Tensor bn_bias = weight.slice(0, index_weight, num_bn_biases + index_weight);
			index_weight += num_bn_biases;
			
			at::Tensor bn_weights = weight.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;
			
			at::Tensor bn_running_mean = weight.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;
			
			at::Tensor bn_running_var = weight.slice(0, index_weight, index_weight + num_bn_biases);
			index_weight += num_bn_biases;

			bn_bias = bn_bias.view_as(bn_imp->bias);
			bn_weights = bn_weights.view_as(bn_imp->weight);
			bn_running_var = bn_running_var.view_as(bn_imp->running_variance);
			bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);

			bn_imp->bias.set_data(bn_bias);
			bn_imp->weight.set_data(bn_weights);
			bn_imp->running_mean.set_data(bn_running_mean);
			bn_imp->running_variance.set_data(bn_running_var);
		}
		else
		{
			int num_conv_biase = conv_imp->bias.numel();
			at::Tensor conv_bias = weight.slice(0, index_weight, index_weight + num_conv_biase);
			index_weight += num_conv_biase;
			conv_bias = conv_bias.view_as(conv_imp->bias);
			conv_imp->bias.set_data(conv_bias);
		}

		int num_weight = conv_imp->weight.numel();
		at::Tensor conv_weight = weight.slice(0, index_weight, index_weight + num_weight);
		index_weight += num_weight;

		conv_weight = conv_weight.view_as(conv_imp->weight);
		conv_imp->weight.set_data(conv_weight);
	}
}

torch::Tensor Darknet::forward(torch::Tensor x){

	int module_count = module_list.size();
	std::vector<torch::Tensor> outputs(module_count);

	torch::Tensor result;
	int write = 0;

	for (int i = 0; i < module_count;++i)
	{
		map<string, string> block = blocks[i + 1];

		string layer_type = block["type"];

		if (layer_type == "net")
			continue;
		if (layer_type == "convolutional" || layer_type == "upsample"){
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl*>(module_list[i].ptr().get());
			x = seq_imp->forward(x);
			outputs[i] = x;
		}
		else if (layer_type=="route")
		{
			int start = std::stoi(block["start"]);
			int end = std::stoi(block["end"]);

			if (start>0) start = start - i;
			if (end==0)
			{
				x = outputs[i +start];
			}
			else
			{
				if (end > 0)end = end - i;
				
				torch::Tensor map_1 = outputs[i + start];
				torch::Tensor map_2 = outputs[i + end];

				x = torch::cat({ map_1, map_2 }, 1);
			}
			outputs[i] = x;
		}
		else if (layer_type=="shortcut")
		{
			int from = std::stoi(block["from"]);
			if (from > 0) from = from - i;

			x = outputs[i + from]+outputs[i-1];
			outputs[i] = x;
		}
		else if (layer_type=="yolo")
		{
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl*>(module_list[i].ptr().get());

			map<string, string> net_info = blocks[0];

			int inp_dim = std::stoi(net_info["height"],0);
			int num_classes = std::stoi(net_info["classes"],0);

			x = seq_imp->forward(x, inp_dim, num_classes, *_device);

			if (write==0)
			{
				result = x;
				write = 1;
			}
			else
			{
				result = torch::cat( { result, x }, 1 );
			}
			outputs[i] = outputs[i - 1];
		}
	}
	return result;
}

torch::Tensor Darknet::write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf /* = 0.4 */)
{
	auto conf_maak = (prediction.select(2, 4) > confidence).to(torch::kFloat32).unsqueeze(2);

	prediction.mul_(conf_maak);

	auto ind_nz = torch::nonzero(prediction.select(2, 4)).transpose(0, 1).contiguous();

	if (ind_nz.size(0)==0)
	{
		return torch::zeros({ 0 });
	}
	//prediction.options() tensor的数据格式级选项 TensorOptions(dtype=float, device=cpu, layout=Strided, requires_grad=false)
	torch::Tensor box_a = torch::ones(prediction.sizes(), prediction.options());
	//top left x =ccenterX-w/2
	box_a.select(2, 0) = prediction.select(2, 0) - prediction.select(2, 2).div(2);
	box_a.select(2, 1) = prediction.select(2, 1) - prediction.select(2, 3).div(2);
	box_a.select(2, 2) = prediction.select(2, 0) + prediction.select(2, 2).div(2);
	box_a.select(2, 3) = prediction.select(2, 1) + prediction.select(2, 3).div(2);

	prediction.slice(2, 0, 4) = box_a.slice(2, 0, 4);

	int batch_size = prediction.size(0);
	int item_attr_size = 5;

	torch::Tensor output = torch::ones({ 1,prediction.size(2) + 1 });

	bool write = false;
	int num = 0;
	for (int i=0;i<batch_size;++i)
	{
		auto image_prediction = prediction[i];
		// get the max classes score at each result max_classes[0] max and max_classes[1] index 0是行1 是列
		std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(image_prediction.slice(1, item_attr_size, item_attr_size + num_classes), 1);
		//score
		auto max_conf = std::get<0>(max_classes);
		//index 行id 

		auto max_conf_score = std::get<1>(max_classes);
		max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);
		max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);
		// shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
		image_prediction = torch::cat({ image_prediction.slice(1, 0, 5), max_conf, max_conf_score }, 1);

		//remove item which object confidence=0 返回不等于0的行id
		auto non_zero_index = torch::nonzero(image_prediction.select(1, 4));
		//根据行id 选出 生成的种类
		auto image_prediction_data = image_prediction.index_select(0, non_zero_index.squeeze()).view({ -1, 7 });

		//get unique classes
		std::vector<torch::Tensor> img_classes;//classesID
		for (int m=0,len=image_prediction.size(0);m<len;++m)
		{
			bool found = false;
			for (int n=0;n<img_classes.size();n++)
			{
				auto ret = (image_prediction[m][6] == img_classes[n]);
				if (torch::nonzero(ret).size(0)>0)
				{
					found = true;
					break;
				}
			}
			if (!found) img_classes.push_back(image_prediction[m][6]);
		}

		for (int k = 0; k < img_classes.size(); k++)
		{
			auto cls = img_classes[k]; //取出一类进行比较
			//选出该类的所有预测
			auto cls_mask = image_prediction_data * (image_prediction_data.select(1, 6) == cls).to(torch::kFloat32).unsqueeze(1);
			//取出所有该类别的结果的行号
			auto class_mask_index = torch::nonzero(cls_mask.select(1, 5)).squeeze();
			//取出所有符合结果的行
			auto image_pred_class = image_prediction_data.index_select(0, class_mask_index).view({ -1, 7 });
			// ascend by confidence 升序排序 分值高的在下低的在上
			// seems that inverse method not work
			//（1）将所有框的得分排序，选中最高分及其对应的框：

			std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1, 4));
			//取出排序的行ID
			auto conf_sort_index = std::get<1>(sort_ret);

			// seems that there is something wrong with inverse method
			// conf_sort_index = conf_sort_index.inverse();

			image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();
			//（2）遍历其余的框，如果和当前最高分框的重叠面积(IOU)大于一定阈值，我们就将框删除。
			//（3）从未处理的框中继续选一个得分最高的，重复上述过程。
			for (int w = 0; w < image_pred_class.size(0);++w)
			{

				int mi = image_pred_class.size(0) - 1 - w;//逆序计算iou 即从最大分数开始计算

				if (mi<=0)
				{
					break;
				}
				//计算Iou 
				auto ious = get_bbox_iou(image_pred_class[mi].unsqueeze(0), image_pred_class.slice(0, 0, mi));
				//过滤掉小于阈值的IOUs
				auto iou_mask = (ious<nms_conf).to(torch::kFloat32).unsqueeze();
				
				image_pred_class.slice(0, 0, mi) = image_pred_class.slice(0, 0, mi)*iou_mask;

				auto non_zero_index = torch::nonzero(image_pred_class.select((1, 4)).squeeze();
				//选出该类别中符合的区域
				image_pred_class = image_pred_class.index_select(0, non_zero_index).view({ -1, 7 });
			}
			torch::Tensor batch_index = torch::ones({ image_pred_class.size(0), 1 }).fill_(i);

			if (!write)
			{
				output = torch::cat({ batch_index, image_pred_class }, 1);
				write = true;
			}
			else
			{
				auto out = torch::cat({ batch_index, image_pred_class }, 1);
				output = torch::cat({ output, out }, 0);
			}

			num += 1;
			
		}
	}
}
