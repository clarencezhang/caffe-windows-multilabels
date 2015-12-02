// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  
  softmax_top_vec_.push_back(&prob_);
  
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  
  int num = prob_.num();
  int dim = prob_.count() / num;

  int prob_channels = prob_.channels();  
  CHECK_EQ(dim, prob_channels) << "output of prob size is NOT the same.";
  
  int label_num = bottom[1]->num();
  int label_size = bottom[1]->channels();

  CHECK_EQ(label_size, prob_channels) << "output of prob size is NOT the same as label size.";

  // disable by multiple labels handling process
  /*
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])],
                     Dtype(FLT_MIN)));
  }
  */
  
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
  	int label_index = 0;
	
  	for (int k = 0; k < label_size; k++)
  	{
  	    if (label[i * label_size + k] == 1)
  	    {
  	    	label_index = k;
  	        break;
  	    }
  	}
	
    loss += -log(max(prob_data[i * dim + label_index], Dtype(FLT_MIN)));
  }
  
  return loss / num;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
    
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  
  const Dtype* label = (*bottom)[1]->cpu_data();
  
  int num = prob_.num();
  int dim = prob_.count() / num;

  int prob_channels = prob_.channels();  
  CHECK_EQ(dim, prob_channels) << "output of prob size is NOT the same.";
  
  int label_num = (*bottom)[1]->num();
  int label_size = (*bottom)[1]->channels();
  
  CHECK_EQ(label_size, prob_channels) << "output of prob size is NOT the same as label size.";

  for (int i = 0; i < num; ++i) {
  	int label_index = 0;
	
  	for (int k = 0; k < label_size; k++)
  	{
  	    if (label[i * label_size + k] == 1)
  	    {
  	    	label_index = k;
  	        break;
  	    }
  	}
	
    bottom_diff[i * dim + label_index] -= 1;
  }
  
  /*
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
  }
  */
  
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
}


INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
