#copy ..\\..\\bin\\MainCaller.exe ..\\..\\bin\\train_net_multi_labels.exe
SET GLOG_logtostderr=1
"../../bin/train_net_multi_labels.exe" lenet_solver.prototxt
echo "train done......"
pause
