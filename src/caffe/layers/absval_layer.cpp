
#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype AbsValLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  //const int count = (*top)[0]->count();
  const int count = bottom[0]->count();  
  const Dtype* bottom_data = bottom[0]->cpu_data();    
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  
  //caffe_abs(count, bottom[0]->cpu_data(), top_data);
  //caffe_abs(count, bottom_data, top_data);
  for (int i = 0; i < count; ++i) {
    top_data[i] = fabs(bottom_data[i]);
  }
  return Dtype(0);
  
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {  
    //const int count = top[0]->count();
    const int count = (*bottom)[0]->count();    
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    
    caffe_div(count, top_data, bottom_data, bottom_diff);
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(AbsValLayer);

}  // namespace caffe
