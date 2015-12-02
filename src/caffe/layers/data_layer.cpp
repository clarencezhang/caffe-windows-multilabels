// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
//#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

int g_item_id = 0;

namespace caffe {

	template <typename Dtype>
	void* DataLayerPrefetch(void* layer_pointer) {
		CHECK(layer_pointer);
		DataLayer<Dtype>* layer = static_cast<DataLayer<Dtype>*>(layer_pointer);
		CHECK(layer);
		Datum datum;
		CHECK(layer->prefetch_data_);
		Dtype* top_data = layer->prefetch_data_->mutable_cpu_data(); //数据
		Dtype* top_label;                                            //标签
		if (layer->output_labels_) {
			top_label = layer->prefetch_label_->mutable_cpu_data();
		}
		const Dtype scale = layer->layer_param_.data_param().scale();
		const int batch_size = layer->layer_param_.data_param().batch_size();
		const int crop_size = layer->layer_param_.data_param().crop_size();
		const bool mirror = layer->layer_param_.data_param().mirror();

		if (mirror && crop_size == 0) {//当前实现需要同时设置mirror和cropsize
			LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
				<< "set at the same time.";
		}
		// datum scales
		const int channels = layer->datum_channels_;
		const int height = layer->datum_height_;
		const int width = layer->datum_width_;
		const int size = layer->datum_size_;
		const Dtype* mean = layer->data_mean_.cpu_data();
		
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			//每一批数据的数量是batchsize，一个循环拉取一张

			// get a blob
			CHECK(layer->iter_);
			CHECK(layer->iter_->Valid());
			datum.ParseFromString(layer->iter_->value().ToString());//利用迭代器拉取下一批数据
			const string& data = datum.data();

			int label_blob_channels = layer->prefetch_label_->channels();
			int label_data_dim = datum.label_size();
			CHECK_EQ(layer->prefetch_label_->channels(), datum.label_size()) << "label size is NOT the same.";
			
			if (crop_size) {//如果需要裁剪  
				CHECK(data.size()) << "Image cropping only support uint8 data";
				int h_off, w_off;
				// We only do random crop when we do training.
				//只是在训练阶段做随机裁剪 
				if (layer->phase_ == Caffe::TRAIN) {
					h_off = layer->PrefetchRand() % (height - crop_size);
					w_off = layer->PrefetchRand() % (width - crop_size);
				} else {//测试阶段固定裁剪
					h_off = (height - crop_size) / 2;
					w_off = (width - crop_size) / 2;
				}
				//怎么感觉下面两种情况的代码是一样的？ 
				if (mirror && layer->PrefetchRand() % 2) {
					// Copy mirrored version
					for (int c = 0; c < channels; ++c) {
						for (int h = 0; h < crop_size; ++h) {
							for (int w = 0; w < crop_size; ++w) {
								int top_index = ((item_id * channels + c) * crop_size + h)
									* crop_size + (crop_size - 1 - w);
								int data_index = (c * height + h + h_off) * width + w + w_off;
								Dtype datum_element =
									static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
								top_data[top_index] = (datum_element - mean[data_index]) * scale;
							}
						}
					}
				} else {//如果不需要裁剪  
					// Normal copy
					//我们优先考虑data()，然后float_data() 
					for (int c = 0; c < channels; ++c) {
						for (int h = 0; h < crop_size; ++h) {
							for (int w = 0; w < crop_size; ++w) {
								int top_index = ((item_id * channels + c) * crop_size + h)
									* crop_size + w;
								int data_index = (c * height + h + h_off) * width + w + w_off;
								Dtype datum_element =
									static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
								top_data[top_index] = (datum_element - mean[data_index]) * scale;
							}
						}
					}
				}
			} else {
				// we will prefer to use data() first, and then try float_data()
				if (data.size()) {
					for (int j = 0; j < size; ++j) {
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[j]));
						top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
					}
				} else {
					for (int j = 0; j < size; ++j) {
						top_data[item_id * size + j] =
							(datum.float_data(j) - mean[j]) * scale;
					}
				}
			}

		
			if (g_item_id++ < 5)
			{
				int label_size = datum.label_size();	
				int image_label = 0;
				for (int j = 0; j < label_size; ++j) {
					if (datum.label(j) == 1)
					{
						image_label = j;
						break;
					}
				}	
				
				char strImgRawDataFile[255] = "";
				sprintf(strImgRawDataFile, "caffe_%s_%05d_%d%s", "train", item_id, image_label, ".txt");
				ofstream fout_image_raw_data(strImgRawDataFile);

				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						int pixel_index = h * height + w;
						Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[pixel_index]));

						char strHexByte[3] = "";
						sprintf(strHexByte, "%02X", (unsigned char)datum_element);
						fout_image_raw_data<<" "<<strHexByte;
					}
					
					fout_image_raw_data<<endl;
				}
				
				fout_image_raw_data<<endl;
				for (int j = 0; j < label_size; ++j) {
					fout_image_raw_data<<datum.label(j);
				}	

				fout_image_raw_data.close();
			}
		
			if (layer->output_labels_) {
				int label_size = datum.label_size();				
				for (int j = 0; j < label_size; ++j) {
					top_label[item_id * label_size + j] = datum.label(j);
				}				
				//top_label[item_id] = datum.label();
			}
			
			// go to the next iter
			layer->iter_->Next();
			if (!layer->iter_->Valid()) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				layer->iter_->SeekToFirst();
			}
		}

		return static_cast<void*>(NULL);
	}

	template <typename Dtype>
	DataLayer<Dtype>::~DataLayer<Dtype>() {
		JoinPrefetchThread();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
			CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
			CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
			if (top->size() == 1) {
				output_labels_ = false;
			} else {
				output_labels_ = true;
			}
			
			// Initialize the leveldb
			leveldb::DB* db_temp;
			leveldb::Options options;
			options.create_if_missing = false;
			options.max_open_files = 100;
			
			LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
			
			leveldb::Status status = leveldb::DB::Open(
				options, this->layer_param_.data_param().source(), &db_temp);
			
			CHECK(status.ok()) << "Failed to open leveldb "
				<< this->layer_param_.data_param().source() << std::endl
				<< status.ToString();
			
			db_.reset(db_temp);
			iter_.reset(db_->NewIterator(leveldb::ReadOptions()));//通过迭代器来操纵leveldb
			iter_->SeekToFirst();
			
			// Check if we would need to randomly skip a few data points
			//是否要随机跳过一些数据
			if (this->layer_param_.data_param().rand_skip()) {
				unsigned int skip = caffe_rng_rand() %
					this->layer_param_.data_param().rand_skip();
				LOG(INFO) << "Skipping first " << skip << " data points.";
				while (skip-- > 0) {
					iter_->Next();
					if (!iter_->Valid()) {
						iter_->SeekToFirst();
					}
				}
			}
			
			// Read a data point, and use it to initialize the top blob.  
			//读取一个数据点，用来初始化topblob。所谓初始化，只要是指reshape。  
			//可以观察到下面iter_调用调用next。所以这次读取只是用来读取出来channels等参数的，不作处理。  
			Datum datum;
			datum.ParseFromString(iter_->value().ToString());//利用迭代器读取第一个数据点 
			
			// image图像数据  
			int crop_size = this->layer_param_.data_param().crop_size();//裁剪大小  
			if (crop_size > 0) {//需要裁剪
				(*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
					datum.channels(), crop_size, crop_size);
				prefetch_data_.reset(new Blob<Dtype>(
					this->layer_param_.data_param().batch_size(), datum.channels(),
					crop_size, crop_size));
			} else {//不需要裁剪  
				(*top)[0]->Reshape(
					this->layer_param_.data_param().batch_size(), datum.channels(),
					datum.height(), datum.width());
				prefetch_data_.reset(new Blob<Dtype>(
					this->layer_param_.data_param().batch_size(), datum.channels(),
					datum.height(), datum.width()));
			}
			
			LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
				<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
				<< (*top)[0]->width();

			/*
			// label标签数据 
			if (output_labels_) {
				(*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
				prefetch_label_.reset(
					new Blob<Dtype>(this->layer_param_.data_param().batch_size(), 1, 1, 1));
			}
			*/
			
			// label标签数据 
			if (output_labels_) {
				(*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), datum.label_size(), 1, 1);
				prefetch_label_.reset(
					new Blob<Dtype>(this->layer_param_.data_param().batch_size(), datum.label_size(), 1, 1));
			}
			
			// datum size
			datum_channels_ = datum.channels();
			datum_height_ = datum.height();
			datum_width_ = datum.width();
			datum_size_ = datum.channels() * datum.height() * datum.width();
			CHECK_GT(datum_height_, crop_size);
			CHECK_GT(datum_width_, crop_size);
			
			// check if we want to have mean  是否要减去均值  
			if (this->layer_param_.data_param().has_mean_file()) {
				const string& mean_file = this->layer_param_.data_param().mean_file();
				LOG(INFO) << "Loading mean file from" << mean_file;
				BlobProto blob_proto;
				ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
				data_mean_.FromProto(blob_proto);
				CHECK_EQ(data_mean_.num(), 1);
				CHECK_EQ(data_mean_.channels(), datum_channels_);
				CHECK_EQ(data_mean_.height(), datum_height_);
				CHECK_EQ(data_mean_.width(), datum_width_);
			} else {
				// Simply initialize an all-empty mean.
				data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
			}
			
			// Now, start the prefetch thread. Before calling prefetch, we make two
			// cpu_data calls so that the prefetch thread does not accidentally make
			// simultaneous cudaMalloc calls when the main thread is running. In some
			// GPUs this seems to cause failures if we do not so.
			prefetch_data_->mutable_cpu_data();
			if (output_labels_) {
				prefetch_label_->mutable_cpu_data();
			}
			
			data_mean_.cpu_data();
			
			DLOG(INFO) << "Initializing prefetch";
			CreatePrefetchThread();
			DLOG(INFO) << "Prefetch initialized.";
	}

	template <typename Dtype>
	void DataLayer<Dtype>::CreatePrefetchThread() {
		phase_ = Caffe::phase();
		const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
			(this->layer_param_.data_param().mirror() ||
			this->layer_param_.data_param().crop_size());
		if (prefetch_needs_rand) {
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		} else {
			prefetch_rng_.reset();
		}
		// Create the thread.
		//CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
		//      static_cast<void*>(this))) << "Pthread execution failed.";
		thread_ = thread(DataLayerPrefetch<Dtype>, reinterpret_cast<void*>(this));
	}

	template <typename Dtype>
	void DataLayer<Dtype>::JoinPrefetchThread() {
		//CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
		thread_.join();
	}

	template <typename Dtype>
	unsigned int DataLayer<Dtype>::PrefetchRand() {
		CHECK(prefetch_rng_);
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		return (*prefetch_rng)();
	}

	template <typename Dtype>
	Dtype DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			// First, join the thread
			// First, join the thread 等待线程结束  
			JoinPrefetchThread();
			
			// Copy the data
			// Copy the data拷贝数据到top，即该层的输出 
			caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
				(*top)[0]->mutable_cpu_data());
			
			if (output_labels_) {
				caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
					(*top)[1]->mutable_cpu_data());
			}
			
			// Start a new prefetch thread
			CreatePrefetchThread();
			return Dtype(0.);
	}

	INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
