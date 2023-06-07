#ifndef VOCABULARY_H
#define VOCABULARY_H

#include "tweet_reader.h"
#include "glove_reader.h"

class Vocabulary {
	public:
		Vocabulary(const WordsFrequencies& words_frequencies, const GloveReader& glove_reader);

		int64_t GetIndex(const std::string& word) const;
		int64_t GetPaddingIndex() const {return static_cast<int64_t>(pad_index_);}
		torch::Tensor GetEmbeddings() const;
		int64_t GetEmbeddingsCount() const {return static_cast<int64_t>(embeddings_.size());}
	private:
		std::unordered_map<std::string, size_t> words_to_index_map_;
		std::vector<torch::Tensor> embeddings_;
		size_t unk_index_;
		size_t pad_index_;
};

#endif //VOCABULARY_H
