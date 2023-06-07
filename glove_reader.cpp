#include "glove_reader.h"

GloveReader::GloveReader(const std::string& file_name, int64_t vec_size) {
	std::ifstream file;
	file.exceptions(std::ifstream::badbit);
	file.open(file_name);
	if(file) {
		auto sizes = {static_cast<long>(vec_size)};
		unknown_ = torch::zeros(sizes, torch::dtype(torch::kFloat));

		std::vector<float> vec(static_cast<size_t>(vec_size));

		std::string line;
		std::string key;
		std::string token;
		while(std::getline(file, line)) {
			if(!line.empty()) {
				std::stringstream line_stream(line);
				size_t num = 0;
				while(std::getline(line_stream, token, ' ')) {
					if(num == 0)
						key = token;
					else
						vec[num - 1] = std::stof(token);
					++num;
				}
				assert(num == (static_cast<size_t>(vec_size) + 1));
				torch::Tensor tvec = torch::from_blob(
						vec.data(),
						sizes,
						torch::TensorOptions().dtype(torch::kFloat)
				);
				dict_[key] = tvec.clone();
			}
		}
	}
}

torch::Tensor GloveReader::Get (const std::string& key) const {
	auto i = dict_.find(key);
	if(i != dict_.end()) {
		return i->second;
	}
	else {
		return torch::empty({0});
	}
}
