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
	//inp_dim ����ͼƬά�� ��/��
	torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, vector<float> anchors,
		int num_classes, torch::Device device) {

		int batch_size = prediction.size(0);//��������
		int stride = floor(inp_dim/prediction.size(2));//���ű���
		int grid_size = floor(inp_dim / stride);//����ÿ������
		int bbox_attrs = 5 + num_classes;//ÿ�����ݸ��� ��������ƫ���� + ���ƫ����+ Ŀ�����=5 + ���������
		int num_anchors = anchors.size() / 2;// anchors ÿ��������ߣ�Ϊһ��

		for (int i=0;i<anchors.size();++i)
		{
			anchors[i] = anchors[i] / stride;//��anchor ���
		}
		//���������ת��Ϊ batch_size x bbox_attrs*num_anchor x grid_size*grid_size ά�ȵ�����
		torch::Tensor result = prediction.view({ batch_size,bbox_attrs*num_anchors,grid_size*grid_size });
		result = result.transpose(1, 2).contiguous();//����ת��
		result = result.view({ batch_size, grid_size*grid_size*num_anchors, bbox_attrs });//����任

		/*
		��������£�YOLO ����Ԥ��߽�����ĵ�ȷ�����ꡣ��Ԥ�⣺
		�� ��Ԥ��Ŀ�������Ԫ���Ͻ���ص�ƫ�ƣ�
		�� ʹ������ͼ��Ԫ��ά�ȣ�1�����й�һ����ƫ�ơ�
		�����ǵ�ͼ��Ϊ����������ĵ�Ԥ���� (0.4, 0.7)���������� 13 x 13 ����ͼ�ϵ������� (6.4, 6.7)����ɫ��Ԫ�����Ͻ������� (6,6),(6,6)��+(0.4, 0.7)����
		���ǣ����Ԥ�⵽�� x,y ������� 1������ (1.2, 0.7)����ô���������� (7.2, 6.7)��ע��������ں�ɫ��Ԫ�Ҳ�ĵ�Ԫ�У���� 7 �еĵ� 8 ����Ԫ��������� YOLO ��������ۣ���Ϊ������Ǽ����ɫ����Ԥ��Ŀ�깷����ô�������ı����ں�ɫ��Ԫ�У���Ӧ�������Աߵ�����Ԫ�С�
		��ˣ�Ϊ�˽��������⣬���Ƕ����ִ�� sigmoid �����������ѹ�������� 0 �� 1 ֮�䣬��Чȷ�����Ĵ���ִ��Ԥ�������Ԫ�С�
		
		Object ������ʾĿ���ڱ߽���ڵĸ��ʡ���ɫ�������������� Object ����Ӧ�ýӽ� 1�������䴦������� Object �������ܽӽ� 0��
		objectness �����ļ���Ҳʹ�� sigmoid ��������������Ա����Ϊ���ʡ�
		*/
		result.select(2, 0).sigmoid_();//���� tx ƫ����
		result.select(2, 1).sigmoid_();//���� ty ƫ����
		result.select(2.4).sigmoid_();//Ŀ����� 


		auto grid_len = torch::arange(grid_size);

		vector<torch::Tensor> args = torch::meshgrid({ grid_len,grid_len });
		torch::Tensor x_offset = args[1].contiguous().view({ -1, 1 });//���һ������
		torch::Tensor y_offset = args[0].contiguous().view({ -1, 1 });

		cout << "x_offset" << x_offset << endl;
		cout << "y_offset" << y_offset << endl;

		x_offset = x_offset.to(device);
		y_offset = y_offset.to(device);

		//����ƴ�ӣ��γ�A=(grid_len * grid_len* grid_len * grid_len���� һ�еľ��� 
		//Ȼ����num_anchors �� ��Ϊ  A�� * num_anchors�� �ľ��� Ȼ�����Ϊ x�� 2�еľ��� x=1/2 * A * num_anchors 

		auto x_y_offset = torch::cat({ x_offset,y_offset },1).repeat({ 1,num_anchors }).view({ -1,2 }).unsqueeze(0);

		result.slice(2, 0, 2).add_(x_y_offset);
		torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), { num_anchors, 2 });

		result.slice(2, 2, 4).exp_().mul_(anchors_tensor);

		/*������Ŷ�
		������Ŷȱ�ʾ��⵽�Ķ�������ĳ�����ĸ��ʣ��繷��è���㽶�������ȣ����� v3 ֮ǰ��YOLO
		��Ҫ��������ִ�� softmax ����������
		���ǣ�YOLO v3 ������������ƣ�����ѡ��ʹ�� sigmoid ��������Ϊ��������ִ�� softmax
		������ǰ��������ǻ���ġ�����֮�������������һ�������ô����ȷ���䲻������һ�����
		�����������ü������ COCO ���ݼ�������ȷ�ġ����ǣ����������Ů�ԡ���Women���͡��ˡ���Person��ʱ��
		�ü��費���С����������ѡ��ʹ�� Softmax �������ԭ��*/
		result.slice(2, 5, 5 + num_classes).sigmoid_();//  5��5+num_classes 
		
		 //������� �������õ� ��ʵ������Ϳ�� ��1��2 �������е�0��4 �����ݳ��� stride��
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
	int prev_filters = 3;//����channel
	std::vector<int> output_filters;
	int index = 0;
	int filters = 0;//conv channel �ò����channel;
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
