#pragma once
void _nams(int *keep_out, int* num_out, const float* boxes_host, int boxes_num,
	int boxes_dim, float nums_overlap_thresh, int device_id);