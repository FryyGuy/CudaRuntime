#pragma once
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>

class Tensor {
private:
	std::vector<size_t> m_dimensions;
	std::vector<float> m_input;
public:
	Tensor(std::string filepath, std::vector<size_t> dims);
	Tensor(std::vector<size_t> dims);
	~Tensor();

	std::vector<size_t> getDims() { return m_dimensions; }
	float* getPointerToInput() { return m_input.data(); }
	size_t size() { return m_input.size(); }

	void assignData(float* input);
};
