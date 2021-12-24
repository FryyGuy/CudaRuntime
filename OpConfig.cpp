#include "OpConfig.hpp"

//==========================================================
// OpConfig C'tor
// Take in a json object and assigns data to relevant fields
//==========================================================
OpConfig::OpConfig(json& j) {

	// store name
	m_name = j["name"];

	// store global dims
	auto globalDims = j["global_dims"];
	for (int i = 0; i < globalDims.size(); i++) {
		m_globalDims.push_back(static_cast<size_t>(globalDims.at(i)));
	}

	// store local dims
	auto localDims = j["local_dims"];
	for (int i = 0; i < localDims.size(); i++) {
		m_localDims.push_back(static_cast<size_t>(localDims.at(i)));
	}

	// store all inputs
	std::vector<std::string> inputFiles = j["inputs"];
	for (int i = 0; i < inputFiles.size(); i++) {
		auto inputFilePath = inputFiles.at(i);
		std::shared_ptr<Tensor> t(new Tensor(inputFilePath, m_globalDims));
		m_inputs.push_back(t);
	}

	// ensure output_dims match output shape and store output reference results for later
	auto refOutputDims = j["ref_output_dims"];
	for (int i = 0; i < refOutputDims.size(); i++) {
		m_outputDims.push_back(static_cast<size_t>(refOutputDims.at(i)));
	}

	std::string refOutputFilePath = j["ref_output"];
	m_referenceOutput = std::shared_ptr<Tensor>(new Tensor(refOutputFilePath, m_outputDims));

	m_output = std::shared_ptr<Tensor>(new Tensor(m_outputDims));

	auto outHeight = m_outputDims[0];
	auto outWidth = m_outputDims[1];
	auto outChannels = m_outputDims[2];

	assert(m_referenceOutput->size() == (outHeight * outWidth * outChannels));
}

OpConfig::~OpConfig() {}

std::shared_ptr<Tensor> OpConfig::getInputTensor(int idx) {
	return m_inputs.at(idx);
}

int OpConfig::inputListSize() {
	return m_inputs.size();
}

int OpConfig::inputTensorSize(int idx) {
	return m_inputs.at(idx)->size();
}

int OpConfig::refOutputTensorSize() {
	return m_referenceOutput->size();
}