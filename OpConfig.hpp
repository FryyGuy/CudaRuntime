#include <vector>
#include <chrono>

#include "Tensor.hpp"
#pragma once

class OpConfig {
private:
	std::vector<std::unique_ptr<Tensor>> inputs;
	std::unique_ptr<Tensor> output;
	std::vector<size_t> globalDims;
	std::vector<size_t> localDims;
	std::chrono::time_point<std::chrono::system_clock> executionTime;
public:

};