#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

using namespace std;
struct Darknet :torch::nn::Module
{
public:
	Darknet(const char*conf_file, torch::Device *device);
	map<string, string>* get_net_info();

	void load_weights(const char *weight_file);

	torch::Tensor forward(torch::Tensor x);

	torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf = 0.4);

private:
	torch::Device *_device;
	vector<map<string, string>>blocks;

	torch::nn::Sequential features;

	vector<torch::nn::Sequential> module_list;

	void load_cfg(const char *cfg_file);

	void create_modules();

	int get_int_from_cfg(map<string, string> block, string key, int default_value);

	string get_string_from_cfg(map<string, string> block, string key, string default_value);

};