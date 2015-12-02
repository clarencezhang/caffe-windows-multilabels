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
				
		}//����ƫ��������Ϊ�������ڴ���һά������ʽ�ģ�������Ҫ����ƫ����������
		
		// Copy from source. 
		// If copy_diff is false, we copy the data; 
		// if copy_diff is true,  we copy the diff.
		// data & diff �洢�ڲ�ͬ��Blob��
		void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
			bool reshape = false);

		//��cpu��������data
		inline Dtype data_at(const int n, const int c, const int h,
			const int w) const {
				return *(cpu_data() + offset(n, c, h, w));
		}

		//��cpu��������diff
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

		// ������data�ŵ�cpu/gpu��, ��ʵ��Щ�������ǵ���SyncedMemory�ĺ���, ���������ݵ�ָ��
		const Dtype* cpu_data() const;
		void set_cpu_data(Dtype* data);
		const Dtype* gpu_data() const;

		// ������diff�ŵ�cpu/gpu��, ��ʵ��Щ�������ǵ���SyncedMemory�ĺ���, ���������ݵ�ָ��
		const Dtype* cpu_diff() const;
		const Dtype* gpu_diff() const;

		// �����ݷ�data/diff��cpu��, ����������cpu��ָ�룬���ı����ݵ�״̬ΪHEAD_AT_CPU 
		// �����ݷ�data/diff��gpu��, ����������gpu��ָ�룬���ı����ݵ�״̬ΪHEAD_AT_GPU 
		Dtype* mutable_cpu_data();
		Dtype* mutable_gpu_data();
		Dtype* mutable_cpu_diff();
		Dtype* mutable_gpu_diff();

		// ����data_�����ݣ����Ǽ�ȥdiff_������
		void Update();

		// ��proto�����ݽ�������ʵ���Ƿ����л�
		void FromProto(const BlobProto& proto);

		// ���������л���proto����
		void ToProto(BlobProto* proto, bool write_diff = false) const;

		// Set the data_/diff_ shared_ptr to point to the SyncedMemory holding the
		// data_/diff_ of Blob other -- useful in layers which simply perform a copy
		// in their forward or backward pass.
		// This deallocates the SyncedMemory holding this blob's data/diff, as
		// shared_ptr calls its destructor when reset with the = operator.
		void ShareData(const Blob& other);
		void ShareDiff(const Blob& other);

	protected:
		shared_ptr<SyncedMemory> data_; // data���ݣ�ָ��SyncedMemory������ָ��
		shared_ptr<SyncedMemory> diff_; // diff��ʾ"��"(lr_scale * gradient)�����ڸ���data_

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

		DISABLE_COPY_AND_ASSIGN(Blob); // �Ѹ���Ŀ������캯���͸�ֵ����������ֹ��
	};  // class Blob

	// Blobs provide a unified memory interface
	
}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
