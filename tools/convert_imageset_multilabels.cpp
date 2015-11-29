// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

//#include <windows.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <time.h>
#include <iostream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

using std::pair;
using std::string;
using std::endl;

// Usage:
//    convert_imageset_multilabels  root_folder list_file leveldb_name


//int main(int argc, char** argv) {
int convert_imageset_multilabels(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  
  if (argc < 4 || argc > 5) {
    printf("Convert a set of images to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 1;
  }
  LOG(INFO) <<"argv[1]: "<<argv[1]<<std::endl;
  LOG(INFO) <<"argv[2]: "<<argv[2]<<std::endl;
  LOG(INFO) <<"argv[3]: "<<argv[3]<<std::endl;

  std::ifstream infile(argv[2]);
  if (!infile)  
  {  
	  fprintf(stderr,"cannot open %s !/n", argv[2]);  
	  return 1;  
  }  

  //std::vector<std::pair<string, int> > lines;
  std::vector<std::pair<string, std::vector<int> > > lines;
  std::string filename;
  int label;

  /*
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  */
  int image_num = 0;
  infile >> image_num;

  while (infile >> filename) {
  	//std::vector<int> labelList(10);
	std::vector<int> labelList;
	labelList.reserve(10);
	for (int i = 0; i < 10; i++)
	{
	    infile >> label;
		labelList.push_back(label);
	}
		
    lines.push_back(std::make_pair(filename, labelList));
  }  
  
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";
  std::cout << "A total of " << lines.size() << " images." << std::endl;
  
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  //options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(options, argv[3], &db);
  //CAFFE_CHECK(status.ok()) << "Failed to open leveldb " << argv[3];
  CAFFE_CHECK(status.ok()); // << "Failed to open leveldb " << argv[3] << ". Is it already existing?";
  
  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;

  int channels = GRAY_CHANNELS;
  
  for (int line_id = 0; line_id < lines.size(); ++line_id) 
  {
    //if (!ReadImageToDatum(root_folder + lines[line_id].first, lines[line_id].second, &datum))
    if (!ReadImageLabelArrayToDatum(lines[line_id].first, lines[line_id].second, channels, &datum))
    {
      continue;
    }
    
    if (!data_size_initialized) 
    {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } 
    else 
    {
      const string& data = datum.data();
      CAFFE_CHECK_EQ(data.size(), data_size); // << "Incorrect data field size " << data.size();
    }
    
    // get the value & sequential
    string value;
    datum.SerializeToString(&value);
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id, lines[line_id].first.c_str());	
    db->Put(leveldb::WriteOptions(), std::string(key_cstr), value);	

	//datum.Clear();
	
	if (++count % 1000 == 0) {
	  std::cout << "Processed " << count << " files." << std::endl;
    }
	
	// disable batch collection
	/*
    batch->Put(string(key_cstr), value);	

	if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      //LOG(INFO) << "Processed " << count << " files."<< std::endl;
	  std::cout << "Processed " << count << " files." << std::endl;
      delete batch;
      batch = new leveldb::WriteBatch();
    }
    */

	/*
    datum.SerializeToString(&value);
	_snprintf(key, kMaxKeyLength, "%08d", itemid);
    db->Put(leveldb::WriteOptions(), std::string(key), value);	
	*/
  }

  /*
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    //LOG(INFO) << "Processed " << count << " files.";
	std::cout << "Processed " << count << " files." << std::endl;
  }
  */

  LOG(INFO) <<"Note!!! " << lines.size() << " images convert to levelDB done...... " << std::endl;
  
  delete batch;
  delete db;
  return 0;
}
