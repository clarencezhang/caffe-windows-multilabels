// Copyright 2014 BVLC and contributors.

//#include <cuda_runtime.h>
//#include <cublas_v2.h>

#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

// 初始化数据成员，智能指针指向SyncedMemory对象。
// 此时SyncedMemory对象其实并没有为自己的“数据”申请内存，只是自己“数据”的大小（size）。
template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  CAFFE_CHECK_GE(num, 0);
  CAFFE_CHECK_GE(channels, 0);
  CAFFE_CHECK_GE(height, 0);
  CAFFE_CHECK_GE(width, 0);
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;

  // reset函数用于停止对保存指针的所有权的共享。共享资源的引用计数减一
  if (count_) {
    data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
  } else {
    data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    diff_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.num(), other.channels(), other.height(), other.width());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CAFFE_CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CAFFE_CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CAFFE_CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CAFFE_CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CAFFE_CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CAFFE_CHECK(data_);
  return reinterpret_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CAFFE_CHECK(data_);
  return reinterpret_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CAFFE_CHECK(diff_);
  return reinterpret_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CAFFE_CHECK(diff_);
  return reinterpret_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CAFFE_CHECK_EQ(count_, other.count());
  data_ = other.data(); // 赋值操作共享other中的资源，并停止对原有资源的共享
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CAFFE_CHECK_EQ(count_, other.count());
  diff_ = other.diff(); // 赋值操作共享other中的资源，并停止对原有资源的共享
}

// y = αx+y, 其中α是标量(-1)，x和y矢量。
// 这里调用是为了实现了两个向量的减法
template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        reinterpret_cast<const Dtype*>(diff_->cpu_data()),
        reinterpret_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    // perform computation on GPU
    /*caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        reinterpret_cast<const Dtype*>(diff_->gpu_data()),
        reinterpret_cast<Dtype*>(data_->mutable_gpu_data()));*/
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

// 从source拷贝数据。copy_diff作为标志来区分是拷贝data还是拷贝diff
template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (num_ != source.num() || channels_ != source.channels() ||
      height_ != source.height() || width_ != source.width()) {
    if (reshape) {
      Reshape(source.num(), source.channels(), source.height(), source.width());
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    /*if (copy_diff) {
      CUDA_CHECK(cudaMemcpy(diff_->mutable_gpu_data(), source.gpu_diff(),
          sizeof(Dtype) * count_, cudaMemcpyDeviceToDevice));
    } else {
      CUDA_CHECK(cudaMemcpy(data_->mutable_gpu_data(), source.gpu_data(),
          sizeof(Dtype) * count_, cudaMemcpyDeviceToDevice));
    }
    break;*/
  case Caffe::CPU:
    if (copy_diff) {
      memcpy(diff_->mutable_cpu_data(), source.cpu_diff(),
          sizeof(Dtype) * count_);
    } else {
      memcpy(data_->mutable_cpu_data(), source.cpu_data(),
        sizeof(Dtype) * count_);
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// 从proto读数据进来，其实就是反序列化
template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
  // 初始化，分配数组空间
  Reshape(proto.num(), proto.channels(), proto.height(), proto.width());
  
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

// 将数据序列化到proto保存
template <typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->set_num(num_);
  proto->set_channels(channels_);
  proto->set_height(height_);
  proto->set_width(width_);
  proto->clear_data();
  proto->clear_diff();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const Dtype* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);

}  // namespace caffe

