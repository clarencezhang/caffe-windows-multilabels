copy ..\\..\\bin\\MainCaller.exe ..\\..\\bin\\test_net.exe
SET GLOG_logtostderr=1
"../../bin/test_net.exe" lenet_test.prototxt lenet_iter_10000 100 CPU
pause
