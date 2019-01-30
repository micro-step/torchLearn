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



	};
}
