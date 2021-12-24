#include "Tensor.hpp"

Tensor::Tensor(std::string filepath, std::vector<size_t> dims) {
	assert(dims.size() == 3);
	m_dimensions = dims;

	std::fstream inputFile(filepath);
	std::string floatNum;

	while (std::getline(inputFile, floatNum, ' ')) {
		m_input.push_back(std::stof(floatNum));
	}

	auto height = m_dimensions[0];
	auto width = m_dimensions[1];
	auto channels = m_dimensions[2];

	assert(m_input.size() == (channels * height * width));
}

Tensor::Tensor(std::vector<size_t> dims) {
	assert(dims.size() == 3);
	m_dimensions = dims;
}

Tensor::~Tensor() {

}

void Tensor::assignData(float* input) {
	auto size = m_dimensions[0] * m_dimensions[1] * m_dimensions[2];
	m_input.assign(input, input + size);
}