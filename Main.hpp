#include <memory>

#include "OpConfig.hpp"

#pragma once

class JsonParser {
public:
	std::unique_ptr<OpConfig> createOpConfig();

};