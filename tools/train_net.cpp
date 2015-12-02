// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)


//lenet_solver.prototxt 

int main_train_net(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  //::google::SetLogDestination(0, "V:\\Caffe\\windows_version\\caffe-windows\\caffe-windows\\examples\\baby_adult\\temp\\");
  ::google::SetLogDestination(0, "D:\\caffe-windows\\caffe-windows\\examples\\mnist\\Log\\");
  
  if (argc < 2 || argc > 3) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 1;
  }
  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

 
  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver.Solve(argv[2]);
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
