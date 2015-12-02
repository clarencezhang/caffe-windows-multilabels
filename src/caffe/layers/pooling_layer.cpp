// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

	template <typename Dtype>
	void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
			CHECK_EQ(top->size(), 1) << "PoolingLayer takes a single blob as output.";
			kernel_size_ = this->layer_param_.pooling_param().kernel_size();//核大小 
			stride_ = this->layer_param_.pooling_param().stride();//步长 
			pad_ = this->layer_param_.pooling_param().pad();
			if (pad_ != 0) {
				CHECK_EQ(this->layer_param_.pooling_param().pool(),
					PoolingParameter_PoolMethod_AVE)
					<< "Padding implemented only for average pooling.";
			}
			channels_ = bottom[0]->channels();//通道 
			height_ = bottom[0]->height();//高  
			width_ = bottom[0]->width();//宽  
			pooled_height_ = static_cast<int>(ceil(static_cast<float>(
				height_ + 2 * pad_ - kernel_size_) / stride_)) + 1; //计算采样之后的高
			pooled_width_ = static_cast<int>(ceil(static_cast<float>(
				width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;	//计算采样之后的宽  
			(*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
				pooled_width_);	//采样之后top大小
				
			// If stochastic pooling, we will initialize the random index part.
			if (this->layer_param_.pooling_param().pool() ==
				PoolingParameter_PoolMethod_STOCHASTIC) {
					rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
						pooled_width_);
			}
	}

	// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
	// case?
	template <typename Dtype>
	Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data(); 	// 采样层输入
			Dtype* top_data = (*top)[0]->mutable_cpu_data();	// 采样层输出
			
			// Different pooling methods. We explicitly do the switch outside the for
			// loop to save time, although this results in more codes.
			int top_count = (*top)[0]->count();
			switch (this->layer_param_.pooling_param().pool()) {
			case PoolingParameter_PoolMethod_MAX:
				// Initialize
				for (int i = 0; i < top_count; ++i) {
					top_data[i] = -FLT_MAX;
				}
				// The main loop
				for (int n = 0; n < bottom[0]->num(); ++n) {
					for (int c = 0; c < channels_; ++c) {
						for (int ph = 0; ph < pooled_height_; ++ph) {
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int hstart = ph * stride_;
								int wstart = pw * stride_;
								int hend = min(hstart + kernel_size_, height_);
								int wend = min(wstart + kernel_size_, width_);

								//找出核范围内最大 
								for (int h = hstart; h < hend; ++h) {
									for (int w = wstart; w < wend; ++w) {
										top_data[ph * pooled_width_ + pw] =
											max(top_data[ph * pooled_width_ + pw],
											bottom_data[h * width_ + w]);
									}
								}
							}
						}
						
						// compute offset
						bottom_data += bottom[0]->offset(0, 1);
						top_data += (*top)[0]->offset(0, 1);
					}
				}
				break;
				
			case PoolingParameter_PoolMethod_AVE:
				for (int i = 0; i < top_count; ++i) {
					top_data[i] = 0;
				}
				// The main loop
				for (int n = 0; n < bottom[0]->num(); ++n) {
					for (int c = 0; c < channels_; ++c) {
						for (int ph = 0; ph < pooled_height_; ++ph) {
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int hstart = ph * stride_ - pad_;
								int wstart = pw * stride_ - pad_;
								int hend = min(hstart + kernel_size_, height_ + pad_);
								int wend = min(wstart + kernel_size_, width_ + pad_);
								int pool_size = (hend - hstart) * (wend - wstart);
								hstart = max(hstart, 0);
								wstart = max(wstart, 0);
								hend = min(hend, height_);
								wend = min(wend, width_);

								// 核范围内算平均
								for (int h = hstart; h < hend; ++h) {
									for (int w = wstart; w < wend; ++w) {
										top_data[ph * pooled_width_ + pw] +=
											bottom_data[h * width_ + w];
									}
								}
								top_data[ph * pooled_width_ + pw] /= pool_size;
							}
						}
						
						// compute offset
						bottom_data += bottom[0]->offset(0, 1);
						top_data += (*top)[0]->offset(0, 1);
					}
				}
				break;
			case PoolingParameter_PoolMethod_STOCHASTIC:
				NOT_IMPLEMENTED;
				break;
			default:
				LOG(FATAL) << "Unknown pooling method.";
			}
			return Dtype(0.);
	}

	template <typename Dtype>
	void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			if (!propagate_down) {
				return;
			}
			const Dtype* top_diff = top[0]->cpu_diff();	// top权值差值diff，用于更新权值
			const Dtype* top_data = top[0]->cpu_data(); // top权值数据
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();	// bottom权值数据	
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();	// bottom权值差值diff，用于更新权值
			
			// Different pooling methods. We explicitly do the switch outside the for
			// loop to save time, although this results in more codes.
			memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
			switch (this->layer_param_.pooling_param().pool()) {
			case PoolingParameter_PoolMethod_MAX:
				// The main loop
				for (int n = 0; n < top[0]->num(); ++n) {
					for (int c = 0; c < channels_; ++c) {
						for (int ph = 0; ph < pooled_height_; ++ph) {
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int hstart = ph * stride_;
								int wstart = pw * stride_;
								int hend = min(hstart + kernel_size_, height_);
								int wend = min(wstart + kernel_size_, width_);
								for (int h = hstart; h < hend; ++h) {
									for (int w = wstart; w < wend; ++w) {
										// 采样层输出的残差传播给输入。由于是最大采样方法，输出存的都是输入范围内最大的值，
										// 所以残差传播的时候也只有范围内最大的值受影响  
										bottom_diff[h * width_ + w] +=
											top_diff[ph * pooled_width_ + pw] *
											(bottom_data[h * width_ + w] ==
											top_data[ph * pooled_width_ + pw]);
									}
								}
							}
						}
						// offset
						bottom_data += (*bottom)[0]->offset(0, 1);
						top_data += top[0]->offset(0, 1);
						bottom_diff += (*bottom)[0]->offset(0, 1);
						top_diff += top[0]->offset(0, 1);
					}
				}
				break;
			case PoolingParameter_PoolMethod_AVE:
				// The main loop
				for (int n = 0; n < top[0]->num(); ++n) {
					for (int c = 0; c < channels_; ++c) {
						for (int ph = 0; ph < pooled_height_; ++ph) {
							for (int pw = 0; pw < pooled_width_; ++pw) {
								int hstart = ph * stride_ - pad_;
								int wstart = pw * stride_ - pad_;
								int hend = min(hstart + kernel_size_, height_ + pad_);
								int wend = min(wstart + kernel_size_, width_ + pad_);
								int pool_size = (hend - hstart) * (wend - wstart);
								hstart = max(hstart, 0);
								wstart = max(wstart, 0);
								hend = min(hend, height_);
								wend = min(wend, width_);
								for (int h = hstart; h < hend; ++h) {
									for (int w = wstart; w < wend; ++w) {
										// 采样层输出的残差传播给输入，由于是平均采样，所以权重都是1 / poolsize
										bottom_diff[h * width_ + w] +=
											top_diff[ph * pooled_width_ + pw] / pool_size;
									}
								}
							}
						}
						// offset
						bottom_data += (*bottom)[0]->offset(0, 1);
						top_data += top[0]->offset(0, 1);
						bottom_diff += (*bottom)[0]->offset(0, 1);
						top_diff += top[0]->offset(0, 1);
					}
				}
				break;
			case PoolingParameter_PoolMethod_STOCHASTIC:
				NOT_IMPLEMENTED;
				break;
			default:
				LOG(FATAL) << "Unknown pooling method.";
			}
	}


	INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
