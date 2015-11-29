// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <ctime>
#include <process.h>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	shared_ptr<Caffe> Caffe::singleton_;

	// curand seeding
	int64_t cluster_seedgen(void) {
		int64_t s, seed, pid;
		pid = _getpid();
		s = time(NULL);
		seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
		return seed;
	}


	Caffe::Caffe()
		: mode_(Caffe::CPU), phase_(Caffe::TRAIN), random_generator_() 
	{
	}

	Caffe::~Caffe() 
	{
	}

	void Caffe::set_random_seed(const unsigned int seed) 
	{
		// RNG seed
		Get().random_generator_.reset(new RNG(seed));
	}


	class Caffe::RNG::Generator {
	public:
		Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
		explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
		caffe::rng_t* rng() { return rng_.get(); }
	private:
		shared_ptr<caffe::rng_t> rng_;
	};

	Caffe::RNG::RNG() : generator_(new Generator) { }

	Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

	Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
		generator_.reset(other.generator_.get());
		return *this;
	}

	void* Caffe::RNG::generator() {
		return static_cast<void*>(generator_->rng());
	}

}  // namespace caffe
