// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

	const float kLOG_THRESHOLD = 1e-20;

	template <typename Dtype>
	void MultinomialLogisticLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CAFFE_CHECK_EQ(bottom.size(), 2); // << "Loss Layer takes two blobs as input.";
			CAFFE_CHECK_EQ(top->size(), 0); // << "Loss Layer takes no output.";
			CAFFE_CHECK_EQ(bottom[0]->num(), bottom[1]->num());
				// << "The data and label should have the same number.";
			CAFFE_CHECK_EQ(bottom[1]->channels(), 1);
			CAFFE_CHECK_EQ(bottom[1]->height(), 1);
			CAFFE_CHECK_EQ(bottom[1]->width(), 1);
	}

	template <typename Dtype>
	Dtype MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* bottom_label = bottom[1]->cpu_data();
			int num = bottom[0]->num();
			int dim = bottom[0]->count() / bottom[0]->num();
			Dtype loss = 0;
			for (int i = 0; i < num; ++i) {
				int label = static_cast<int>(bottom_label[i]);
				Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
				loss -= log(prob);
			}
			return loss / num;
	}

	template <typename Dtype>
	void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();
			const Dtype* bottom_label = (*bottom)[1]->cpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			int num = (*bottom)[0]->num();
			int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
			memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
			for (int i = 0; i < num; ++i) {
				int label = static_cast<int>(bottom_label[i]);
				Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
				bottom_diff[i * dim + label] = -1. / prob / num;
			}
	}


	template <typename Dtype>
	void InfogainLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CAFFE_CHECK_EQ(bottom.size(), 2); // << "Loss Layer takes two blobs as input.";
			CAFFE_CHECK_EQ(top->size(), 0); // << "Loss Layer takes no output.";
			CAFFE_CHECK_EQ(bottom[0]->num(), bottom[1]->num());
				// << "The data and label should have the same number.";
			CAFFE_CHECK_EQ(bottom[1]->channels(), 1);
			CAFFE_CHECK_EQ(bottom[1]->height(), 1);
			CAFFE_CHECK_EQ(bottom[1]->width(), 1);
			BlobProto blob_proto;
			ReadProtoFromBinaryFile(this->layer_param_.infogain_loss_param().source(),
				&blob_proto);
			infogain_.FromProto(blob_proto);
			CAFFE_CHECK_EQ(infogain_.num(), 1);
			CAFFE_CHECK_EQ(infogain_.channels(), 1);
			CAFFE_CHECK_EQ(infogain_.height(), infogain_.width());
	}


	template <typename Dtype>
	Dtype InfogainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* bottom_label = bottom[1]->cpu_data();
			const Dtype* infogain_mat = infogain_.cpu_data();
			int num = bottom[0]->num();
			int dim = bottom[0]->count() / bottom[0]->num();
			CAFFE_CHECK_EQ(infogain_.height(), dim);
			Dtype loss = 0;
			for (int i = 0; i < num; ++i) {
				int label = static_cast<int>(bottom_label[i]);
				for (int j = 0; j < dim; ++j) {
					Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
					loss -= infogain_mat[label * dim + j] * log(prob);
				}
			}
			return loss / num;
	}

	template <typename Dtype>
	void InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();
			const Dtype* bottom_label = (*bottom)[1]->cpu_data();
			const Dtype* infogain_mat = infogain_.cpu_data();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			int num = (*bottom)[0]->num();
			int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
			CAFFE_CHECK_EQ(infogain_.height(), dim);
			for (int i = 0; i < num; ++i) {
				int label = static_cast<int>(bottom_label[i]);
				for (int j = 0; j < dim; ++j) {
					Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
					bottom_diff[i * dim + j] = - infogain_mat[label * dim + j] / prob / num;
				}
			}
	}


	template <typename Dtype>
	void EuclideanLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CAFFE_CHECK_EQ(bottom.size(), 2); // << "Loss Layer takes two blobs as input.";
			//CAFFE_CHECK_EQ(top->size(), 0); // << "Loss Layer takes no as output.";
            CAFFE_CHECK_LE(top->size(), 1); // << modified to accomodate the regression case with multiple lables
            CAFFE_CHECK_EQ(bottom[0]->num(), bottom[1]->num());
				// << "The data and label should have the same number.";
			CAFFE_CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
			CAFFE_CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CAFFE_CHECK_EQ(bottom[0]->width(), bottom[1]->width());
            
			difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());

            // EuclideanLossLayer allows computing loss accuracy for each sample item in TEST mode. 
            // This time, we will need to initialize the top blob.
            if (top->size() > 0)
            {
                //(*top)[0]->Reshape(bottom[0]->num(), 1, 1, 1);
                (*top)[0]->Reshape(1, 1, 1, 1);
            }
	}

	template <typename Dtype>
	Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();
            int dim = bottom[0]->count() / bottom[0]->num();

			caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
				difference_.mutable_cpu_data());
            
			Dtype loss = caffe_cpu_dot(
				count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);

            // During TEST mode, compute edclidean loss accuracy and record them every iteration
            if (top->size() > 0)
            {
                #if 0
				Dtype* top_loss = (*top)[0]->mutable_cpu_data();
				Dtype* diff_data = difference_.mutable_cpu_data();
                
    			for (int i = 0; i < num; ++i) {
    				// loss
    				double item_loss = 0.0;    				
    				for (int j = 0; j < dim; ++j) {
    					item_loss += sqrt(diff_data[i*dim + j] * diff_data[i*dim + j]);
    				}
                    top_loss[i] = Dtype(item_loss);
    			}
                #endif

				Dtype* diff_data = difference_.mutable_cpu_data();
                Dtype accuracy = 0;
                Dtype face_bouding_box_length = 39.0;
                
    			for (int i = 0; i < num; ++i) {
    				double dist_error = 0.0;
                    bool detect_flag = true;
                    
    				for (int j = 0; j < dim/2; ++j) {                        
    					dist_error = sqrt((diff_data[i*dim+j*2] * diff_data[i*dim+j*2]) + (diff_data[i*dim+j*2+1] * diff_data[i*dim+j*2+1]));

                        if (dist_error/face_bouding_box_length > 0.05)
                        {
                            detect_flag = false;
                            break;
                        }
    				}

                    if (detect_flag)
                    {
                        ++accuracy;
                    }                    
    			}

                (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
            }

			return loss;
	}

	template <typename Dtype>
	void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			int count = (*bottom)[0]->count();
			int num = (*bottom)[0]->num();
			// Compute the gradient
			caffe_cpu_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
				(*bottom)[0]->mutable_cpu_diff());
	}

	template <typename Dtype>
	void AccuracyLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
			CAFFE_CHECK_EQ(bottom.size(), 2); // << "Accuracy Layer takes two blobs as input.";
			CAFFE_CHECK_EQ(top->size(), 1); // << "Accuracy Layer takes 1 output.";
			
			CAFFE_CHECK_EQ(bottom[0]->num(), bottom[1]->num());
				// << "The data and label should have the same number.";
			// disable by multiple label channels
			//CAFFE_CHECK_EQ(bottom[1]->channels(), 1);
			CAFFE_CHECK_EQ(bottom[1]->height(), 1);
			CAFFE_CHECK_EQ(bottom[1]->width(), 1);
			
			(*top)[0]->Reshape(1, 2, 1, 1);
	}

	template <typename Dtype>
	Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
		
			Dtype accuracy = 0;
			Dtype logprob = 0;

			// bottom[0] - prob, bottom[1] - label
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* bottom_label = bottom[1]->cpu_data();
			
			int num = bottom[0]->num();
			int dim = bottom[0]->count() / bottom[0]->num();

			int prob_size = bottom[0]->channels();  
			CAFFE_CHECK_EQ(dim, prob_size); // << "output of prob(bottom[0]) size is NOT the same.";

			int label_num = bottom[1]->num();
			int label_size = bottom[1]->channels();

			CAFFE_CHECK_EQ(label_size, prob_size); // << "output of prob size is NOT the same as label size.";

			// disable by multiple channel labels handling process
			/*
			for (int i = 0; i < num; ++i) {
				// Accuracy
				Dtype maxval = -FLT_MAX;
				int max_id = 0;
				for (int j = 0; j < dim; ++j) {
					if (bottom_data[i * dim + j] > maxval) {
						maxval = bottom_data[i * dim + j];
						max_id = j;
					}
				}
				
				if (max_id == static_cast<int>(bottom_label[i])) {
					++accuracy;
				}
				
				Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
					Dtype(kLOG_THRESHOLD));
				logprob -= log(prob);
			}
			*/

			for (int i = 0; i < num; ++i) {
				// Accuracy
				Dtype maxval = -FLT_MAX;
				int max_id = 0;
				int label_index = 0;
				
				for (int j = 0; j < dim; ++j) {
					if (bottom_data[i * dim + j] > maxval) {
						maxval = bottom_data[i * dim + j];
						max_id = j;
					}
				}

			  	for (int k = 0; k < label_size; k++)
			  	{
			  	    if (bottom_label[i * label_size + k] == 1)
			  	    {
			  	    	label_index = k;
			  	        break;
			  	    }
			  	}	
				
				if (max_id == label_index) {
					++accuracy;
				}
				
				Dtype prob = max(bottom_data[i * dim + label_index], Dtype(kLOG_THRESHOLD));
				logprob -= log(prob);
			}
	
			// LOG(INFO) << "Accuracy: " << accuracy;
			(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
			(*top)[0]->mutable_cpu_data()[1] = logprob / num;
			
			// Accuracy layer should not be used as a loss function.
			return Dtype(0);
	}

	template <typename Dtype>
	void HingeLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			CAFFE_CHECK_EQ(bottom.size(), 2); // << "Hinge Loss Layer takes two blobs as input.";
			CAFFE_CHECK_EQ(top->size(), 0); // << "Hinge Loss Layer takes no output.";
	}

	template <typename Dtype>
	Dtype HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* label = bottom[1]->cpu_data();
			int num = bottom[0]->num();
			int count = bottom[0]->count();
			int dim = count / num;

			caffe_copy(count, bottom_data, bottom_diff);
			for (int i = 0; i < num; ++i) {
				bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
			}
			for (int i = 0; i < num; ++i) {
				for (int j = 0; j < dim; ++j) {
					bottom_diff[i * dim + j] = max(Dtype(0), 1 + bottom_diff[i * dim + j]);
				}
			}
			return caffe_cpu_asum(count, bottom_diff) / num;
	}

	template <typename Dtype>
	void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			const Dtype* label = (*bottom)[1]->cpu_data();
			int num = (*bottom)[0]->num();
			int count = (*bottom)[0]->count();
			int dim = count / num;

			caffe_cpu_sign(count, bottom_diff, bottom_diff);
			for (int i = 0; i < num; ++i) {
				bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
			}
			caffe_scal(count, Dtype(1. / num), bottom_diff);
	}

	INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
	INSTANTIATE_CLASS(InfogainLossLayer);
	INSTANTIATE_CLASS(EuclideanLossLayer);
	INSTANTIATE_CLASS(AccuracyLayer);
	INSTANTIATE_CLASS(HingeLossLayer);

}  // namespace caffe