// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

	template <typename Dtype>
	void SoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 1) << "Softmax Layer takes a single blob as input.";
			CHECK_EQ(top->size(), 1) << "Softmax Layer takes a single blob as output.";
			
			// 输出分配空间，初始化
			(*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			
			// sum_multiplier_ 这里都是1，用于辅助计算，可以看作一个列向量，或者行数为1的矩阵
			sum_multiplier_.Reshape(1, bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
			for (int i = 0; i < sum_multiplier_.count(); ++i) {
				multiplier_data[i] = 1.;
			}
			
			// 临时变量scale_分配空间，大小为num,可以看作一个行向量  
			scale_.Reshape(bottom[0]->num(), 1, 1, 1);
	}

	template <typename Dtype>
	Dtype SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = (*top)[0]->mutable_cpu_data();
			Dtype* scale_data = scale_.mutable_cpu_data();
			
			// 把输出看成是num层，每层dim个元素 
			int num = bottom[0]->num();
			int dim = bottom[0]->count() / bottom[0]->num();
			memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());
			
			// we need to subtract the max to avoid numerical issues, compute the exp,
			// and then normalize.
			// 找出每一层的最大值 
			for (int i = 0; i < num; ++i) {
				scale_data[i] = bottom_data[i*dim];
				for (int j = 0; j < dim; ++j) {
					scale_data[i] = max(scale_data[i], bottom_data[i * dim + j]);
				}
			}
			
			// subtraction  通过矩阵相乘的方式来计算，有num层的top_data，每层元素减去该层的最大值。太巧妙了
			// cblas_sgemm - C = alpha*op( A )*op( B ) + beta*C 
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
				scale_data, sum_multiplier_.cpu_data(), 1., top_data);
			
			// Perform exponentiation 计算自然对数 
			caffe_exp<Dtype>(num * dim, top_data, top_data);
			
			// sum after exp - 每一层各自求和放到scale_data中  
			// y = A * x，则取alpha=1.0，beta=0.0, 
			// A = top_data(num * dim), x = sum_multiplier_(1 * dim), y = scale_data(num * 1)
			caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
				sum_multiplier_.cpu_data(), 0., scale_data);
			
			// Do division - 每一层各自除以该层的和 
			for (int i = 0; i < num; ++i) {
				caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], top_data + i * dim);
			}
			return Dtype(0);
	}

	template <typename Dtype>
	void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			Dtype* scale_data = scale_.mutable_cpu_data();

			// 把输出看成是num层，每层dim个元素 
			int num = top[0]->num();
			int dim = top[0]->count() / top[0]->num();
			memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());
			
			// Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
			for (int i = 0; i < num; ++i) {
				scale_data[i] = caffe_cpu_dot<Dtype>(dim, top_diff + i * dim,
					top_data + i * dim); //每一层，top_diff和top_data计算内积 
			}
			
			// subtraction 每一层bottom_diff的元素减去该层的对应的内积  
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
				scale_data, sum_multiplier_.cpu_data(), 1., bottom_diff);
			
			// elementwise multiplication 元素各自相乘  
			caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
	}


	INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
