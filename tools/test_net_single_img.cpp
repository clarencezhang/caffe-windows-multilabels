// 测试单张图片

//#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "opencvlib.h"
#include "caffe/caffe.hpp"


using namespace caffe;  // NOLINT(build/namespaces)


// 4个参数
// [net prototxt]		V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\mnist\lenet_test_single_img.prototxt
// [net parameters]		V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\mnist\lenet_iter_86000
// [iterations]			1
// [CPU/GPU]			CPU
int main_test_net_single_img(int argc, char** argv) {
	if (argc < 4 || argc > 6) {
		LOG(INFO) << "test_net net_proto pretrained_net_proto iterations "
			<< "[CPU/GPU] [Device ID]";
		return 1;
	}

	Caffe::set_phase(Caffe::TEST);

	if (argc >= 5 && strcmp(argv[4], "GPU") == 0) 
	{
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		if (argc == 6) {
			device_id = atoi(argv[5]);
		}
		//Caffe::SetDevice(device_id);
		LOG(INFO) << "Using GPU #" << device_id;
	} else {
		LOG(INFO) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	Net<float> caffe_test_net(argv[1]);
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);

	// song add
	const vector<boost::shared_ptr<Layer<float> > > &layers = caffe_test_net.layers();
	MemoryDataLayer<float>* memoryDataLayer = static_cast<MemoryDataLayer<float>*>(&(*layers[0])); 
	
	cv::Mat img = cv::imread("test_img\\3568-3900-5792-8635_1_13.jpg", IMREAD_GRAYSCALE);
	cv::resize(img, img, cv::Size(28, 28));
	img.convertTo(img, CV_32F, 1.0/255);
	//cv::imshow("img", img);
	//cv::waitKey(0);

	float *bottom = (float*)img.data;
	float topValue = 2;
	float *top = &topValue;
	int n = 1;
	memoryDataLayer->Reset(bottom, top, n);
	

	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	const float * pResult = result[1]->cpu_data();
	int num = result[1]->num();
	int dim = result[1]->count() / result[1]->num();

	float maxval = -FLT_MAX;
	int max_id = 0;
	for(int j=0; j<dim; ++j) {
		if(pResult[j] > maxval)
		{
			maxval = pResult[j];
			max_id = j;
		}
		LOG(INFO) << j << " prob:" << pResult[j] << std::endl;
	}
	
	

	/*int total_iter = atoi(argv[3]);
	LOG(INFO) << "Running " << total_iter << " iterations.";

	double test_accuracy = 0;
	for (int i = 0; i < total_iter; ++i) {
		const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
		test_accuracy += result[0]->cpu_data()[0];
		LOG(INFO) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
	}
	test_accuracy /= total_iter;
	LOG(INFO) << "Test accuracy: " << test_accuracy;*/

	return 0;
}
