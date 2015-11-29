// This file is used to replace the glog library. <by yangsong>

#ifndef CHECK_NUM_HPP_
#define CHECK_NUM_HPP_

#include <iostream>

using namespace std;

using std::cout;
using std::endl;

/*
template<typename T> inline bool CAFFE_CHECK(T condition)
{
	if(condition == 0)
	{
		cout << "CAFFE_CHECK failed." << endl;
		return false;
	}
	else
	{
		return true;
	}
}
*/
template<typename T> inline bool CAFFE_CHECK(T condition)
{
	if(condition == 0)
	{
		cout << "CAFFE_CHECK failed." << endl;
		return false;
	}
	else
	{
		return true;
	}
}


inline bool CAFFE_DCHECK(bool condition)
{
	if(!condition)
	{
		cout << "CAFFE_DCHECK error." << endl;
	}
	return condition;
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_GE(Dtype num1, Dtype2 num2)
{
	if(num1 >= num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_GE error." << endl;
		return false;
	}
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_GT(Dtype num1, Dtype2 num2)
{
	if(num1 > num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_GT error." << endl;
		return false;
	}
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_LE(Dtype num1, Dtype2 num2)
{
	if(num1 <= num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_LE error." << endl;
		return false;
	}
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_LT(Dtype num1, Dtype2 num2)
{
	if(num1 < num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_LT error." << endl;
		return false;
	}
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_EQ(Dtype num1, Dtype2 num2)
{
	if(num1 == num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_EQ error." << endl;
		return false;
	}
}

template <typename Dtype, typename Dtype2> inline bool CAFFE_CHECK_NE(Dtype num1, Dtype2 num2)
{
	if(num1 != num2)
	{
		return true;
	}
	else
	{
		cout << "CAFFE_CHECK_NE error." << endl;
		return false;
	}
}




#endif // CHECK_NUM_HPP_