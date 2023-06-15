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

	// torch::Tensor packed_text, packed_length;
	// std::tie(packed_text, packed_length) = torch::_pack_padded_sequence(embedded, length.squeeze(1), false);
	torch::nn::utils::rnn::PackedSequence packed_seq = torch::nn::utils::rnn::pack_padded_sequence(
		embedded,
		length.squeeze(1),
		false
	);

	// auto rnn_out = rnn_->forward(packed_text, packed_length);
	auto rnn_out = rnn_->forward_with_packed_input(packed_seq);

	// auto hidden_state = rnn_out.state.narrow(0, 0, 1);
	auto hidden_state = std::get<0>(rnn_out).data().narrow(0, 0, 1);
	hidden_state.squeeze(0);

	auto last_index = rnn_->options.num_layers() - 2;
	hidden_state = at::cat({
		hidden_state.narrow(0, last_index, 1).squeeze(0),
		hidden_state.narrow(0, last_index + 1, 1).squeeze(0)
	}, 1);

	auto hidden = dropout_(hidden_state);

	return fc_(hidden);
}

void SentimentRNNImpl::SetPretrainedEmbeddings(const at::Tensor& weights) {
	torch::NoGradGuard guard;
	embeddings_weights_.copy_(weights);
}
