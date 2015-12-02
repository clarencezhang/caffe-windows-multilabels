// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class Blob {
	public:
		Blob()
			: num_(0), channels_(0), height_(0), width_(0), count_(0), data_(),
			diff_() {}
		explicit Blob(const int num, const int channels, const int height,
			const int width);
		
		void Reshape(const int num, const int channels, const int height,
			const int width);
		void ReshapeLike(const Blob& other);
		
		inline int num() const { return num_; }
		inline int channels() const { return channels_; }
		inline int height() const { return height_; }
		inline int width() const { return width_; }
		inline int count() const { return count_; }

		// The conventional blob dimensions for data are number N x channel K x height H x width W. 
		// Blob memory is row-major in layout so the last / rightmost dimension changes fastest. 
		// For example, the value at index (n, c, h, w) is physically located at index ((n * K + c) * H + h) * W + w.
		inline int offset(const int n, const int c = 0, const int h = 0,
			const int w = 0) const {
				CHECK_GE(n, 0);
				CHECK_LE(n, num_);
				CHECK_GE(channels_, 0);
				CHECK_LE(c, channels_);
				CHECK_GE(height_, 0);
				CHECK_LE(h, height_);
				CHECK_GE(width_, 0);
				CHECK_LE(w, width_);
				return ((n * channels_ + c) * height_ + h) * width_ + w;
				
		}//计算偏移量，因为数据在内存是一维数组形式的，所以需要计算偏移量来访问
		
		// Copy from source. 
		// If copy_diff is false, we copy the data; 
		// if copy_diff is true,  we copy the diff.
		// data & diff 存储在不同的Blob中
		void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
			bool reshape = false);

		//从cpu访问数据data
		inline Dtype data_at(const int n, const int c, const int h,
			const int w) const {
				return *(cpu_data() + offset(n, c, h, w));
		}

		//从cpu访问数据diff
		inline Dtype diff_at(const int n, const int c, const int h,
			const int w) const {
				return *(cpu_diff() + offset(n, c, h, w));
		}

		inline const shared_ptr<SyncedMemory>& data() const {
			CHECK(data_);
			return data_;
		}

		inline const shared_ptr<SyncedMemory>& diff() const {
			CHECK(diff_);
			return diff_;
		}

		// 把数据data放到cpu/gpu上, 其实这些函数就是调用SyncedMemory的函数, 并返回数据的指针
		const Dtype* cpu_data() const;
		void set_cpu_data(Dtype* data);
		const Dtype* gpu_data() const;

		// 把数据diff放到cpu/gpu上, 其实这些函数就是调用SyncedMemory的函数, 并返回数据的指针
		const Dtype* cpu_diff() const;
		const Dtype* gpu_diff() const;

		// 把数据放data/diff到cpu上, 返回数据在cpu的指针，并改变数据的状态为HEAD_AT_CPU 
		// 把数据放data/diff到gpu上, 返回数据在gpu的指针，并改变数据的状态为HEAD_AT_GPU 
		Dtype* mutable_cpu_data();
		Dtype* mutable_gpu_data();
		Dtype* mutable_cpu_diff();
		Dtype* mutable_gpu_diff();

		// 更新data_的数据，就是减去diff_的数据
		void Update();

		// 从proto读数据进来，其实就是反序列化
		void FromProto(const BlobProto& proto);

		// 将数据序列化到proto保存
		void ToProto(BlobProto* proto, bool write_diff = false) const;

		// Set the data_/diff_ shared_ptr to point to the SyncedMemory holding the
		// data_/diff_ of Blob other -- useful in layers which simply perform a copy
		// in their forward or backward pass.
		// This deallocates the SyncedMemory holding this blob's data/diff, as
		// shared_ptr calls its destructor when reset with the = operator.
		void ShareData(const Blob& other);
		void ShareDiff(const Blob& other);

	protected:
		shared_ptr<SyncedMemory> data_; // data数据，指向SyncedMemory的智能指针
		shared_ptr<SyncedMemory> diff_; // diff表示"差"(lr_scale * gradient)，用于更新data_

		// a blob is a 4-dimensional array that stores things 
		// in the order of (Num, Channels, Height and Width), 
		// from major to minor, and stored in a C-contiguous fashion.

		//For a convolution layer with 96 filters of 11 x 11 spatial dimension and 3 inputs 
		// the blob is 96 x 3 x 11 x 11. 

		// For an inner product / fully-connected layer with 1000 output channels and 1024 input channels 
		// the parameter blob is 1 x 1 x 1000 x 1024.
		
		int num_;      // N / batch_size, the batch size of the data. 
		               // Batch processing achieves better throughput for communication and device processing.
		int channels_; // channels / K is the feature dimension e.g. for RGB images K = 3.
		int height_;
		int width_;
		
		int count_;    // count_ = num_ * channels_ * height_ * width_;

		DISABLE_COPY_AND_ASSIGN(Blob); // 把该类的拷贝构造函数和赋值操作符给禁止掉
	};  // class Blob

	// Blobs provide a unified memory interface
	
}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
