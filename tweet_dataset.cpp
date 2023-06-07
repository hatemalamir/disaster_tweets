#include "tweet_dataset.h"
#include "tweet.h"

TweetSample TweetDataset::get(size_t index) {
	torch::Tensor target;
	const Tweet* tweet{nullptr};
	if(index < reader_->GetPosSize()) {
		tweet = &reader_->GetPos(index);
		target = torch::tensor(1.f, torch::dtype(torch::kFloat).device(device_).requires_grad(false));
	}
	else {
		tweet = &reader_->GetNeg(index - reader_->GetNegSize());
		target = torch::tensor(0.f, torch::dtype(torch::kFloat).device(device_).requires_grad(false));
	}
	
	// text encoding
	std::vector<int64_t> indices(reader_->GetMaxSize());
	size_t i = 0;
	for(auto& w: tweet->GetWords()) {
		indices[i] = vocabulary_->GetIndex(w);
		++i;
	}

	// padding
	for(; i < indices.size(); ++i) 
		indices[i] = vocabulary_->GetPaddingIndex();

	auto data = torch::from_blob(
		indices.data(),
		{static_cast<int64_t>(reader_->GetMaxSize())},
		torch::dtype(torch::kLong).requires_grad(false)
	);

	auto data_len = torch::tensor(
		static_cast<int64_t>(tweet->size()),
		torch::dtype(torch::kLong).requires_grad(false)
	);

	return {{data.clone().to(device_), data_len.clone()}, target.squeeze()};
}
