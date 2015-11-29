//#include "../../tools/train_net.cpp"
//#include "../../tools/test_net.cpp"
//#include "../../examples/mnist/mnist/convert_mnist_data.cpp"
//#include "../../tools/finetune_net.cpp"
//#include "../../tools/net_speed_benchmark.cpp"
//#include "../../tools/dump_network.cpp"

//#include "../../tools/convert_imageset.cpp"
//#include "../../tools/extract_features.cpp"
//#include "../../tools/convert_imageset.cpp"
//#include "../../tools/compute_image_mean.cpp"

#include <string.h>
//#include <cuda.h>
//#include <curand.h>
//#include <driver_types.h>  // cuda driver types

extern int main_train_net(int argc, char** argv) ;
extern int main_test_net(int argc, char** argv) ;
extern int main_convert_mnist(int argc, char** argv) ;
extern int main_test_net_single_img(int argc, char** argv) ;

extern int convert_imageset_multilabels(int argc, char** argv) ;

int main(int argc, char** argv) 
{
	/*
	//查询GPU的设备 并且获取相应的属性
	cudaDeviceProp prop;
	memset(&prop,0,sizeof(cudaDeviceProp));
	prop.major=1;
	prop.minor=3;
	int count=0;
	int dev;

	int ierr = cudaChooseDevice(&dev,&prop);

	ierr =cudaGetDeviceCount(&count);
	for(int i=0; i<count; i++)
	{
		cudaGetDeviceProperties(&prop,i);
	}
	*/


	// part1  convert the mnist data function
	// V:\Caffe\windows_version\caffe-windows\samples\mnist\t10k-images.idx3-ubyte  V:\Caffe\windows_version\caffe-windows\samples\mnist\t10k-labels.idx1-ubyte   V:\Caffe\windows_version\caffe-windows\samples\mnist\mnist-test-leveldb
	// V:\Caffe\windows_version\caffe-windows\samples\mnist\train-images.idx3-ubyte  V:\Caffe\windows_version\caffe-windows\samples\mnist\train-labels.idx1-ubyte  V:\Caffe\windows_version\caffe-windows\samples\mnist\mnist-train-leveldb
    // main_convert_mnist( argc, argv);


	// part2  train the data function
	// V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\mnist\lenet_solver.prototxt
	main_train_net( argc, argv);


	// part3  test the trained net
	// V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\mnist\lenet_test.prototxt V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\mnist\lenet_iter_86000  100  CPU
	//main_test_net( argc,  argv);


	// part4  run the trained net with single image
	//main_test_net_single_img( argc, argv );

	// part5 convert images with multiple labels into leveldb
	// Usage:
	//    convert_imageset_multilabels  root_folder   list_file   leveldb_name
	//convert_imageset_multilabels( argc, argv );

}