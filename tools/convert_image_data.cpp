// Copyright 2014 BVLC and contributors.
//
// This script converts the MNIST dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include "stdio.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/caffe.hpp"

#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp> 

using namespace cv;

//void convert_Face_dataset(const char* pos_filename,const char* neg_filename, const char* db_filename, int label_count, int channel = 1, int width = 24, int height = 24);

static uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

static void GetID_FromFileName(const char *filename, char *filepath, char *id)
{
    char   drive[_MAX_DRIVE];   
    char   subdir[_MAX_PATH];   
    char   extension[_MAX_PATH];   
    _splitpath(filename,drive,subdir,id,extension);   
    strcpy(filepath, drive);
    strcat(filepath, subdir);
}

void resizeImageAndPutBuffer(Mat &srcImage,char *pixels, int channel, int rows, int cols)
{
	Mat dstImage;
	if (srcImage.channels() == 1 && srcImage.rows == rows && srcImage.cols == cols)
	{

	}
	else
	{
		if (channel == 1 && srcImage.channels() == 3)
			cvtColor(srcImage, dstImage, CV_RGB2GRAY);
		else
			dstImage = srcImage.clone();
		resize(dstImage, srcImage, Size(rows, cols));
	}


	for (int i = 0; i < rows;i++)
	{
		for (int j = 0; j < cols; j++)
		{
			pixels[i*cols + j] = srcImage.at<char>(i, j);
		}
	}
}

void randomBigData(vector<string> &bigData)
{
	srand(unsigned(time(NULL)));
	vector<string>::iterator it = bigData.begin();
	int OneRandomNumber = 10000;

	while (true)
	{
		if (bigData.end() - it < OneRandomNumber )
		{
			break;
		}
		vector<string>::iterator ed = it + OneRandomNumber;

		random_shuffle(it, ed);

		it = ed;
	}
}

//D:\Face\CNN_FaceDetection\pos\pos11.txt  D:\neg_img.txt V:\Caffe\windows_version\caffe-windows\caffe-windows\examples\face_verification\database_test_zhijun\

void convert_Face_dataset(const char *strSubType, const char* list_filename, const char* db_filename, int label_count, int channel = 1, int width = 24, int height = 24)
{
	uint32_t rows;
	uint32_t cols;

	rows = height;
	cols = width;

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << db_filename
		// << ". Is it already existing?";
	if (!status.ok())
	{
        std::cout << "Failed to open leveldb " << db_filename << ". Is it already existing?";
	    return;
	}

	char label;
	char* pixels = new char[rows * cols];
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];
	std::string value;

	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);

	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

    char posFolder[255];
    char posFileListName[255];
    memset(posFolder, 0, sizeof(posFolder));
    memset(posFileListName, 0, sizeof(posFileListName));
    GetID_FromFileName(list_filename, posFolder, posFileListName);

    char dbFolder[255];
    char dbFileName[255];
    memset(dbFolder, 0, sizeof(dbFolder));
    memset(dbFileName, 0, sizeof(dbFileName));
    GetID_FromFileName(db_filename, dbFolder, dbFileName);

	char googleLogDir[255] = "";
	sprintf(googleLogDir, "%sLog", dbFolder);
	::google::SetLogDestination(0, googleLogDir);

    
	// Open list files
	std::ifstream posImage_file(list_filename, std::ios::in);
    if (!posImage_file)
    {
        std::cout << "Unable to open file " << list_filename;
    }

	//CAFFE_CHECK(posImage_file); // << "Unable to open file " << list_filename;

	std::string line;
	vector<string> posPathList;

	while (getline(posImage_file, line))
	{
		if (line.length() == 0)
		{
			continue;
		}
		posPathList.push_back(line);
	}

	posImage_file.close();
	printf("\n\n[%s], list Face Num : %d\n", strSubType, posPathList.size());

#if 0
	randomBigData(posPathList);
	//srand(unsigned(time(NULL)));
	//random_shuffle(posPathList.begin(), posPathList.end());

	printf("random_shuffle pos face ok!\n");
#endif

#if 0
	int posFaceNum = itemid;
	LOG(INFO) << "Neg Face itemid Num " << itemid <<std::endl;

	// Open test files
	std::ifstream negImage_file(test_filename, std::ios::in);
    if (!negImage_file)
    {
        std::cout << "Unable to open file " << test_filename;
    }
    
	//CAFFE_CHECK(negImage_file); // << "Unable to open file " << neg_filename;

	vector<string> negPathList;
	while (getline(negImage_file, line))
	{
		negPathList.push_back(line);
	}
	negImage_file.close();
	printf("neg Face Num %d\n", negPathList.size());
    
#if 0    
	randomBigData(negPathList);
	//random_shuffle(negPathList.begin(), negPathList.end());

	printf("random_shuffle neg face ok!\n");
#endif
#endif

	printf("                                                                                                                     \r");

    char strFirstImageFile[255] = "";
    sprintf(strFirstImageFile, "%s%s.jpg", posFolder, posPathList[0].c_str());            
	Mat firstImage = imread(strFirstImageFile, 0);
	if (firstImage.empty())
	{
        std::cout << "Unable to open image file " << strFirstImageFile;
        LOG(ERROR) << "Unable to open image file " << strFirstImageFile;
		return;
	}

    if (firstImage.rows != rows || firstImage.cols != cols)
    {
        std::cout << "+++[ " << strSubType << " ]+++ image size is invalid, rows : " << rows << ", cols : " << cols << std::endl;
        LOG(ERROR) << "+++[ " << strSubType << " ]+++ image size is invalid, rows : " << rows << ", cols : " << cols << std::endl;
        system("pause");
		return;
    }


    int itemid = 0;    
    
	for (int i = 0; i < posPathList.size(); i++)
	{
		//if (i<posPathList.size())
		{
			printf("[%s] [%d / %d]  is processing\n", strSubType, i, posPathList.size());
            LOG(INFO) << "[" << strSubType << "] [" << i << " / " << posPathList.size() << "  is processing " << std::endl;
            
			line = posPathList[i];
            
            char strImageFile[255] = "";
            sprintf(strImageFile, "%s%s.jpg", posFolder, line.c_str());            
			Mat posImage = imread(strImageFile, 0);
			if (posImage.empty())
			{
                std::cout << "Unable to open image file " << strImageFile;
                LOG(ERROR) << "Unable to open image file " << strImageFile;
				continue;
			}

            char strLabelFile[255] = "";
            sprintf(strLabelFile, "%s%s.txt", posFolder, line.c_str());  
        	std::ifstream label_file(strLabelFile, std::ios::in);
            if (!label_file)
            {
                std::cout << "Unable to open label file " << label_file;
                LOG(ERROR) << "Unable to open label file " << label_file;
                continue;
            }

    		datum.clear_label();
    		datum.clear_data();
        
			resizeImageAndPutBuffer(posImage, pixels, channel, rows, cols);			
			datum.set_data(pixels, rows*cols);
            
            int label = 0;
            int label_item = 0;
            for (int k = 0; k < label_count & !label_file.eof(); k++)
            {
                label_file >> label;
                //datum.set_label(label);
			    datum.add_label(label);
                label_item++;
            }

            if (label_item != label_count)
            {
                printf("Item[%s], label_item[%d] NOT equal label_count[%d] \n", line.c_str(), label_item, label_count);
                continue;
            }
            
			datum.SerializeToString(&value);
			_snprintf(key, kMaxKeyLength, "%08d", itemid++);
			db->Put(leveldb::WriteOptions(), std::string(key), value);
		}
		
#if 0
		//再导入负样本		
		if (i<negPathList.size())
		{
			printf("neg:%d  %dis processing\n",i,negPathList.size());
			line = negPathList[i];
			Mat negImage = imread(line, 0);
			if (negImage.empty())
				continue;
			resizeImageAndPutBuffer(negImage, pixels, channel, rows, cols);
			label = 0;  //neg face label
			datum.set_data(pixels, rows*cols);
			//datum.set_label(label);
			datum.add_label(label);
			datum.SerializeToString(&value);
			_snprintf(key, kMaxKeyLength, "%08d", itemid++);
			db->Put(leveldb::WriteOptions(), std::string(key), value);
		}
#endif		
	}
	
	
	LOG(INFO) << "levelDB : " << db_filename << " generation DONE......" << std::endl;

	delete db;
	delete pixels;

	return;
}

#if 0
void convert_Face_dataset(const char* pos_filename,const char* neg_filename, const char* db_filename, int label_count, int channel = 1, int width = 24, int height = 24)
{
	uint32_t rows;
	uint32_t cols;

	rows = height;
	cols = width;

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << db_filename
		// << ". Is it already existing?";
	if (!status.ok())
	{
        std::cout << "Failed to open leveldb " << db_filename << ". Is it already existing?";
	    return;
	}

	char label;
	char* pixels = new char[rows * cols];
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];
	std::string value;

	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);

    LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

    char posFolder[255];
    char posFileListName[255];
    memset(posFolder, 0, sizeof(posFolder));
    memset(posFileListName, 0, sizeof(posFileListName));
    GetID_FromFileName(pos_filename, posFolder, posFileListName);

    char negFolder[255];
    char negFileListName[255];
    memset(negFolder, 0, sizeof(negFolder));
    memset(negFileListName, 0, sizeof(negFileListName));
    GetID_FromFileName(neg_filename, negFolder, negFileListName);
    
	// Open files
	std::ifstream posImage_file(pos_filename, std::ios::in);
    if (!posImage_file)
    {
        std::cout << "Unable to open file " << pos_filename;
    }

	//CAFFE_CHECK(posImage_file); // << "Unable to open file " << pos_filename;

	std::string line;
	int itemid = 0;
	vector<string> posPathList;

	while (getline(posImage_file,line))
	{
		posPathList.push_back(line);
	}

	posImage_file.close();
	printf("pos Face Num %d\n", posPathList.size());

	randomBigData(posPathList);
	//srand(unsigned(time(NULL)));
	//random_shuffle(posPathList.begin(), posPathList.end());

	printf("random_shuffle pos face ok!\n");
	
	int posFaceNum = itemid;
	LOG(INFO) << "Neg Face itemid Num " << itemid <<std::endl;

	// Open files
	std::ifstream negImage_file(neg_filename, std::ios::in);
    if (!negImage_file)
    {
        std::cout << "Unable to open file " << neg_filename;
    }
    
	//CAFFE_CHECK(negImage_file); // << "Unable to open file " << neg_filename;

	vector<string> negPathList;
	while (getline(negImage_file, line))
	{
		negPathList.push_back(line);
	}
	negImage_file.close();
	printf("neg Face Num %d\n", negPathList.size());
	randomBigData(negPathList);
	//random_shuffle(negPathList.begin(), negPathList.end());

	printf("random_shuffle neg face ok!\n");

	printf("                                                                                                                     \r");
	
	for (int i = 0; i < MAX(negPathList.size(),posPathList.size()); i++)
	{
		// 先导入正样本
		
		if (i<posPathList.size())
		{
			printf("pos:%d  %d  is processing\n",i,posPathList.size());
			line = posPathList[i];
			Mat posImage = imread(line, 0);
			if (posImage.empty())
				continue;
			resizeImageAndPutBuffer(posImage, pixels, channel, rows, cols);
			label = 1;  //POS face label
			datum.set_data(pixels, rows*cols);
			//datum.set_label(label);
			datum.add_label(label);
			datum.SerializeToString(&value);
			_snprintf(key, kMaxKeyLength, "%08d", itemid++);
			db->Put(leveldb::WriteOptions(), std::string(key), value);
		}
		

		//再导入负样本
		
		if (i<negPathList.size())
		{
			printf("neg:%d  %dis processing\n",i,negPathList.size());
			line = negPathList[i];
			Mat negImage = imread(line, 0);
			if (negImage.empty())
				continue;
			resizeImageAndPutBuffer(negImage, pixels, channel, rows, cols);
			label = 0;  //neg face label
			datum.set_data(pixels, rows*cols);
			//datum.set_label(label);
			datum.add_label(label);
			datum.SerializeToString(&value);
			_snprintf(key, kMaxKeyLength, "%08d", itemid++);
			db->Put(leveldb::WriteOptions(), std::string(key), value);
		}
		
	}
	
	
	LOG(INFO) << "neg Face itemid Num " << negPathList .size()<< std::endl;

	delete db;
	delete pixels;
}
#endif



void convert_Multi_dataset(vector<string> &vecfile_list, const char* db_filename,int width = 24, int height = 24, int channel = 1)
{

	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;


	rows = height;
	cols = width;

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << db_filename
		// << ". Is it already existing?";

	char label;
	char* pixels = new char[rows * cols];
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];
	std::string value;

	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

	// Open files
    vector < vector<string>  > vMultiLabelPathList;
    for (int i = 0; i < vecfile_list.size(); i++)
    {
        vector<string > name_list;
        std::ifstream ifsImage_file(vecfile_list[i], std::ios::in);

	    //CAFFE_CHECK(ifsImage_file); // << "Unable to open file " << vecfile_list[i];
        std::string line;
	    int itemid = 0;

        while (getline(ifsImage_file,line))
        {
            name_list.push_back(line);
        }
        ifsImage_file.close();
	    printf("pos Face Num %d\n", vecfile_list[i].size());
        randomBigData(name_list);
        printf("random_shuffle pos face ok!\n");

        vMultiLabelPathList.push_back(name_list);
    }


    int nMaxNum= 0;
    int nIDNum = vMultiLabelPathList.size();
    for (int i = 0; i < vMultiLabelPathList.size(); i++)
    {
        if (vMultiLabelPathList[i].size()>nMaxNum)
        {
            nMaxNum = vMultiLabelPathList[i].size();
        }
    }

	printf("                                                                                                                     \r");
	
    int itemid=0;
    //交替写入样本进去，以得到较好的训练结果。
	for (int i = 0; i < nMaxNum; i++)
	{
        for (int m = 0; m < nIDNum; m++)
        {
            printf("pos:%d_%d  %d_%d is processing\r",i,nMaxNum,m,nIDNum);
            if (m>=vMultiLabelPathList[i].size())
            {
                continue;
            }
            string line = vMultiLabelPathList[i][m];
			Mat Image = imread(line, 0);
			if (Image.empty())
				continue;
			resizeImageAndPutBuffer(Image, pixels, channel, rows, cols);

    		datum.clear_label();
    		datum.clear_data();
    		datum.clear_float_data();
        
            label = i;  //POS face label
			datum.set_data(pixels, rows*cols);
			//datum.set_label(label);
			datum.add_label(label);
			datum.SerializeToString(&value);
			_snprintf(key, kMaxKeyLength, "%08d", itemid++);
			db->Put(leveldb::WriteOptions(), std::string(key), value);
        }
	}

	delete db;
	delete pixels;
}

void Convert_multi(string file_list, const char* db_filename,int width = 24, int height = 24, int channel = 1)
{
    vector<string> vecfile_list;
    std::ifstream ifs(file_list, std::ios::in);
    //CAFFE_CHECK(ifs); // << "Unable to open file " << file_list;
    std::string line;
    int itemid = 0;

    while (getline(ifs,line))
    {
        vecfile_list.push_back(line);
    }
    ifs.close();
    convert_Multi_dataset(vecfile_list, db_filename,width , height, channel);

}

static void convert_dataset(const char* image_filename, const char* label_filename, const char* db_filename)
{
	// Open files
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	//CAFFE_CHECK(image_file); // << "Unable to open file " << image_filename;
	//CAFFE_CHECK(label_file); // << "Unable to open file " << label_file;
	// Read the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CAFFE_CHECK_EQ(magic, 2051); // << "Incorrect image file magic.";
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CAFFE_CHECK_EQ(magic, 2049); // << "Incorrect label file magic.";


	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CAFFE_CHECK_EQ(num_items, num_labels);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << db_filename
		// << ". Is it already existing?";

	char label;
	char* pixels = new char[rows * cols];
	const int kMaxKeyLength = 10;
	char key[kMaxKeyLength];
	std::string value;

	caffe::Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
	for (int itemid = 0; itemid < num_items; ++itemid) {
		image_file.read(pixels, rows * cols);
		label_file.read(&label, 1);

		datum.clear_label();
		datum.clear_data();
		datum.clear_float_data();
            
		datum.set_data(pixels, rows*cols);
		//datum.set_label(label);
		datum.add_label(label);
		
	//	printf("label %c\n",label);

		datum.SerializeToString(&value);
		_snprintf(key, kMaxKeyLength, "%08d", itemid);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
	}

	delete db;
	delete pixels;
}


//V:\Caffe\windows_version\caffe-windows\samples\mnist\train-images.idx3-ubyte  V:\Caffe\windows_version\caffe-windows\samples\mnist\train-labels.idx1-ubyte  V:\Caffe\windows_version\caffe-windows\samples\mnist\train-images_labels.db
void convert_minist_dataset(const char* image_filename, const char* label_filename,
					 const char* db_filename) {
						 // Open files
						 std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
						 std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
						 //CAFFE_CHECK(image_file); // << "Unable to open file " << image_filename;
						 //CAFFE_CHECK(label_file); // << "Unable to open file " << label_file;
						 // Read the magic and the meta data
						 uint32_t magic;
						 uint32_t num_items;
						 uint32_t num_labels;
						 uint32_t rows;
						 uint32_t cols;

						 image_file.read(reinterpret_cast<char*>(&magic), 4);
						 magic = swap_endian(magic);
						 CAFFE_CHECK_EQ(magic, 2051); // << "Incorrect image file magic.";
						 label_file.read(reinterpret_cast<char*>(&magic), 4);
						 magic = swap_endian(magic);
						 CAFFE_CHECK_EQ(magic, 2049); // << "Incorrect label file magic.";
						 image_file.read(reinterpret_cast<char*>(&num_items), 4);
						 num_items = swap_endian(num_items);
						 label_file.read(reinterpret_cast<char*>(&num_labels), 4);
						 num_labels = swap_endian(num_labels);
						 CAFFE_CHECK_EQ(num_items, num_labels);
						 image_file.read(reinterpret_cast<char*>(&rows), 4);
						 rows = swap_endian(rows);
						 image_file.read(reinterpret_cast<char*>(&cols), 4);
						 cols = swap_endian(cols);

						 // Open leveldb
						 leveldb::DB* db;
						 leveldb::Options options;
						 options.create_if_missing = true;
						 options.error_if_exists = true;
						 leveldb::Status status = leveldb::DB::Open(
							 options, db_filename, &db);
						 CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << db_filename
							 // << ". Is it already existing?";

						 char label;
						 char* pixels = new char[rows * cols];
						 const int kMaxKeyLength = 10;
						 char key[kMaxKeyLength];
						 std::string value;

						 caffe::Datum datum;
						 datum.set_channels(1);
						 datum.set_height(rows);
						 datum.set_width(cols);
						 LOG(INFO) << "A total of " << num_items << " items.";
						 LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
						 for (int itemid = 0; itemid < num_items; ++itemid) {
							 printf("%d is reading\n",itemid);
							 image_file.read(pixels, rows * cols);
							 label_file.read(&label, 1);

                    		datum.clear_label();
                    		datum.clear_data();
                    		datum.clear_float_data();
        
							 datum.set_data(pixels, rows*cols);
							 //datum.set_label(label);
							 datum.add_label(label);
							 datum.SerializeToString(&value);
							 //Allen modify
							 _snprintf_s(key, kMaxKeyLength, "%08d", itemid);

							 db->Put(leveldb::WriteOptions(), std::string(key), value);
						 }

						 delete db;
						 delete pixels;
}

//D:\Project\caffe-windows\examples\facedetect\pos_img.txt  D:\Project\caffe-windows\examples\facedetect\neg_img.txt D:\Project\caffe-windows\examples\facedetect\face_train_db

char *Level1_Points_SubFolder[] = {

	"Level1_F1",  
	"Level1_EN1",  
	"Level1_NM1"
};

char *Level2_Points_SubFolder[] = {
	"Level2_LE21",  
	"Level2_LE22",  
	"Level2_RE21",  
	"Level2_RE22",  
	"Level2_N21",   
	"Level2_N22",   
	"Level2_LM21",  
	"Level2_LM22",  
	"Level2_RM21",  
	"Level2_RM22"
};

// D:\Project\caffe-windows\examples\mnist\train-images\train-images.idx3-ubyte D : \Project\caffe - windows\examples\mnist\train-images\train - labels.idx1 - ubyte  D : \Project\caffe - windows\examples\mnist\train - images\train_tt
int convert_image_facial_points(int argc, char** argv) {
	if (argc != 4) {
		printf("This script converts the MNIST dataset to the leveldb format used\n"
			"by caffe to perform classification.\n"
			"Usage:\n"
			"    convert_mnist_data input_image_file input_label_file "
			"output_db_file\n"
			"The MNIST dataset could be downloaded at\n"
			"    http://yann.lecun.com/exdb/mnist/\n"
			"You should gunzip them after downloading.\n");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		//convert_Face_dataset(argv[1], argv[2], atoi(argv[3]), 1, 48, 48);

        //convert_Face_dataset(argv[1], argv[2], atoi(argv[3]), 1, 39, 39);
		//convert_dataset(argv[1], argv[2], argv[3]);

		{
            // train
            char strTrainFolder[255] = "F:\\caffe-windows\\samples\\pose\\caffe_train_test_samples\\train_set\\";
            char strTrainLevel2Folder[255] = "F:\\caffe-windows\\samples\\pose\\caffe_train_test_samples\\train_set\\Level2\\";
            char strTrainListFile[255] = "";
            char strTrainDbFile[255] = "";

            // Level1
            int level1_size = sizeof(Level1_Points_SubFolder) / sizeof(char *);
            int width, height, label_count;
			char strSubType[255] = "";
            
            for (int k = 1; k < level1_size; k++)
            {
                sprintf(strTrainListFile, "%s%s\\list_shuffle.txt", strTrainFolder, Level1_Points_SubFolder[k]);
                sprintf(strTrainDbFile, "%s%s_train.db", strTrainFolder, Level1_Points_SubFolder[k]);

                if (k == 0)
                {
                    // Level1_F1
                    width = 39;
                    height = 39;
                    label_count = 10;
                }
                else
                {
                    // Level1_EN1
                    // Level1_NM1
                    width = 39;
                    height = 31;  
                    label_count = 6;
                }

				sprintf(strSubType, "Train : %s", Level1_Points_SubFolder[k]);
                convert_Face_dataset(strSubType, strTrainListFile, strTrainDbFile, label_count, 1, width, height);
            }

            // Level2
            int level2_size = sizeof(Level2_Points_SubFolder) / sizeof(char *);
            
            for (int k = 0; k < level2_size; k++)
            {
                sprintf(strTrainListFile, "%s%s\\list_shuffle.txt", strTrainLevel2Folder, Level2_Points_SubFolder[k]);
                sprintf(strTrainDbFile, "%s%s_train.db", strTrainLevel2Folder, Level2_Points_SubFolder[k]);

                // Level2
                width = 15;
                height = 15;  
                label_count = 2;

				sprintf(strSubType, "Train : %s", Level2_Points_SubFolder[k]);
                convert_Face_dataset(strSubType, strTrainListFile, strTrainDbFile, label_count, 1, width, height);
            }
		}


        {
            // test
            char strTestFolder[255] = "F:\\caffe-windows\\samples\\pose\\caffe_train_test_samples\\test_set\\";
            char strTestLevel2Folder[255] = "F:\\caffe-windows\\samples\\pose\\caffe_train_test_samples\\test_set\\Level2\\";
            char strTestListFile[255] = "";
            char strTestDbFile[255] = "";

            // Level1
            int level1_size = sizeof(Level1_Points_SubFolder) / sizeof(char *);
            int width, height, label_count;
			char strSubType[255] = "";
            

            for (int k = 1; k < level1_size; k++)
            {
                sprintf(strTestListFile, "%s%s\\list_shuffle.txt", strTestFolder, Level1_Points_SubFolder[k]);
                sprintf(strTestDbFile, "%s%s_test.db", strTestFolder, Level1_Points_SubFolder[k]);

                if (k == 0)
                {
                    // Level1_F1
                    width = 39;
                    height = 39;
                    label_count = 10;
                }
                else
                {
                    // Level1_EN1
                    // Level1_NM1
                    width = 39;
                    height = 31;  
                    label_count = 6;
                }

				sprintf(strSubType, "Test : %s", Level1_Points_SubFolder[k]);
                convert_Face_dataset(strSubType, strTestListFile, strTestDbFile, label_count, 1, width, height);
            }

            // Level2
            int level2_size = sizeof(Level2_Points_SubFolder) / sizeof(char *);
            
            for (int k = 0; k < level2_size; k++)
            {
                sprintf(strTestListFile, "%s%s\\list_shuffle.txt", strTestLevel2Folder, Level2_Points_SubFolder[k]);
                sprintf(strTestDbFile, "%s%s_test.db", strTestLevel2Folder, Level2_Points_SubFolder[k]);

                // Level2
                width = 15;
                height = 15;  
                label_count = 2;

				sprintf(strSubType, "Test : %s", Level2_Points_SubFolder[k]);
                convert_Face_dataset(strSubType, strTestListFile, strTestDbFile, label_count, 1, width, height);
            }
                
        }


        system("pause");  


	}
	return 0;
}
