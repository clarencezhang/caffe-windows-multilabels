// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// Theoretically, CaffeMallocHost and CaffeFreeHost should simply call the
// cudaMallocHost and cudaFree functions in order to create pinned memory.
// However, those codes rely on the existence of a cuda GPU (I don't know
// why that is a must since allocating memory should not be accessing the
// GPU resorce, but it just creates an error as of Cuda 5.0) and will cause
// problem when running on a machine without GPU. Thus, we simply define
// these two functions for safety and possible future change if the problem
// of calling cuda functions disappears in a future version.
//
// In practice, although we are creating unpinned memory here, as long as we
// are constantly accessing them the memory pages almost always stays in
// the physical memory (assuming we have large enough memory installed), and
// does not seem to create a memory bottleneck here.

inline void CaffeMallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
}

inline void CaffeFreeHost(void* ptr) {
  free(ptr);
}


class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();

  // 表示数据的四种状态，未初始化，数据在cpu中，数据在gpu中，数据在cpu和gpu中都有
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; } 
  size_t size() { return size_; }

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory); // 把该类的拷贝构造函数和赋值操作符给禁止掉
};  // class SyncedMemory

// Assuming that data are on the CPU initially, and we have a blob.
//const Dtype* foo;
//Dtype* bar;
//foo = blob.gpu_data(); // data copied cpu->gpu.
//foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
//bar = blob.mutable_gpu_data(); // no data copied.
// ... some operations ...
//bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
//foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
//foo = blob.gpu_data(); // no data copied since both have up-to-date contents
//bar = blob.mutable_cpu_data(); // still no data copied.
//bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
//bar = blob.mutable_cpu_data(); // data copied gpu->cpu.

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
