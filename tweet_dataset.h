#ifndef TWEET_DATASET_H
#define TWEET_DATASET_H

#include "tweet_reader.h"
#include "vocabulary.h"
#include <torch/torch.h>

using TweetData = std::pair<torch::Tensor, torch::Tensor>;
using TweetSample = torch::data::Example<TweetData, torch::Tensor>;

class TweetDataset: public torch::data::Dataset<TweetDataset, TweetSample> {
	public:
		TweetDataset(TweetReader* reader, Vocabulary* vocabulary, torch::DeviceType device) : device_(device), reader_(reader), vocabulary_(vocabulary) {}
		TweetSample get(size_t index) override;
		torch::optional<size_t> size() const override{return reader_->GetPosSize() + reader_->GetNegSize();}
	private:
		torch::DeviceType device_{torch::DeviceType::CPU};
		TweetReader* reader_{nullptr};
		Vocabulary* vocabulary_{nullptr};
};

#endif //TWEET_DATASET_H
