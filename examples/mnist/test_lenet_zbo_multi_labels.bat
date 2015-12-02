#copy ..\\..\\bin\\MainCaller.exe ..\\..\\bin\\test_net_multi_labels.exe
SET GLOG_logtostderr=1
"../../bin/test_net_multi_labels.exe" lenet_test_multi_labels.prototxt lenet_iter_10000 100 CPU
pause
