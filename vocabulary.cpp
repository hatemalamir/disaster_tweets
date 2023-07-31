#include "vocabulary.h"

Vocabulary::Vocabulary(const WordsFrequencies& words_frequencies, const GloveReader& glove_reader) {
	words_to_index_map_.reserve(words_frequencies.size());
	embeddings_.reserve(words_frequencies.size());

	unk_index_ = 0;
	pad_index_ = unk_index_ + 1;

	embeddings_.push_back(glove_reader.GetUnknown());
	
	size_t index = pad_index_ + 1;
	for(auto& wf: words_frequencies) {
		auto embedding = glove_reader.Get(wf.first);
		if(embedding.size(0) != 0) {
			embeddings_.push_back(embedding);
			words_to_index_map_.insert({wf.first, index});
			++index;
		}
		else
			words_to_index_map_.insert({wf.first, unk_index_});
	}
}

int64_t Vocabulary::GetIndex(const std::string& word) const {
	auto i = words_to_index_map_.find(word);
	if(i != words_to_index_map_.end())
		return static_cast<int64_t>(i->second);
	else
		return static_cast<int64_t>(unk_index_);

}

at::Tensor Vocabulary::GetEmbeddings() const {
	at::Tensor weights = torch::stack(embeddings_);
	return weights;
}

