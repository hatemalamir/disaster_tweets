#ifndef GLOVE_READER_H
#define GLOVE_READER_H

#include <string>
#include <torch/torch.h>
#include <unordered_map>

class GloveReader {
	public:
		GloveReader(const std::string& file_name, int64_t vec_size);
		torch::Tensor Get(const std::string& key) const;
		torch::Tensor GetUnknown() const {return unknown_;}
	private:
		torch::Tensor unknown_;
		std::unordered_map<std::string, torch::Tensor> dict_;
};

#endif //GLOVE_READER_H
