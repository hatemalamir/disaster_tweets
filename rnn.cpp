#include "rnn.h"

/////////////////////// SentimentRNNImpl ///////////////////////
SentimentRNNImpl::SentimentRNNImpl(
	int64_t vocab_size,
	int64_t embedding_dim,
	int64_t hidden_dim,
	int64_t output_dim,
	int64_t n_layers,
	bool bidirectional,
	double dropout,
	int64_t pad_idx
): pad_idx_() {
	embeddings_weights_ = register_parameter(
		"embeddings_weights",
		torch::empty({vocab_size, embedding_dim})
	);

	rnn_ = torch::nn::LSTM(
		torch::nn::LSTMOptions(embedding_dim, hidden_dim)
		.num_layers(n_layers)
		.bidirectional(bidirectional)
		.dropout(dropout)
	);
	register_module("rnn", rnn_);

	fc_ = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim * 2, output_dim));
	register_module("fc", fc_);

	dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(dropout));
	register_module("dropout", dropout_);
}

torch::Tensor SentimentRNNImpl::forward(const torch::Tensor& text, const at::Tensor& length) {
	auto embedded = dropout_(torch::embedding(embeddings_weights_, text, pad_idx_));

	auto rnn_out = rnn_->forward(embedded);

	auto hidden_state = std::get<0>(rnn_out);
	auto hidden_state_size = hidden_state.sizes();

	int64_t sequence_len = hidden_state_size[0];
	auto last_hidden_state = hidden_state.narrow(0, sequence_len - 2, 1).squeeze(0);

	auto hidden = dropout_(last_hidden_state);

	auto output = fc_(hidden);

	return output;
}

void SentimentRNNImpl::SetPretrainedEmbeddings(const at::Tensor& weights) {
	torch::NoGradGuard guard;
	embeddings_weights_.copy_(weights);
}
