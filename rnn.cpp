#include "rnn.h"

PackedLSTMImpl::PackedLSTMImpl(const torch::nn::LSTMOptions& options) {
	rnn_ = torch::nn::LSTM(options);
	register_module("rnn", rnn_);
}

/*
std::vector<torch::Tensor> PackedLSTMImpl::flat_weights() const {
	std::vector<torch::Tensor> flat;
	const auto num_directions = rnn_->options.bidirectional() ? 2 : 1;
	for(int64_t layer=0; layer < rnn_->options.num_layers(); layer++) {
		for(auto direction=0; direction < num_directions; direction++) {
			const auto layer_idx = static_cast<size_t>((layer * num_directions) + direction);
			flat.push_back(rnn_->w_ih[layer_idx]);
			flat.push_back(rnn_->w_hh[layer_idx]);
			if(rnn_->options.with_bias_) {
				flat.push_back(rnn_->b_ih[layer_idx]);
				flat.push_back(rnn_->b_hh[layer_idx]);
			}
		}
	}

	return flat;
	return rnn->flat_weights_;
}
*/

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> PackedLSTMImpl::forward(const torch::Tensor& input, const at::Tensor& lengths, torch::Tensor state) {
	if(!state.defined()) {
		const auto max_batch_size = lengths[0].item().toLong();
		const auto num_directions = rnn_->options.bidirectional() ? 2 : 1;
		state = torch::zeros({
				2,
				rnn_->options.num_layers() * num_directions,
				max_batch_size,
				rnn_->options.hidden_size()
			},
			input.options()
		);
	}

	torch::Tensor output, hidden_state, cell_state;
	std::tie(output, hidden_state, cell_state) = torch::lstm(
		input,
		lengths,
		{state[0], state[1]},
		rnn_->flat_weights(),
		rnn_->options.with_bias_,
		rnn_->options.num_layers(),
		rnn_->options.dropout(),
		rnn_->is_training(),
		rnn_->options.bidirectional()
	);
	return {
		output,
		torch::stack({
			hidden_state,
			cell_state
		})
	};
}

const torch::nn::LSTMOptions& options() {
	return rnn_->options;
}

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

	rnn_ = PackedLSTM(
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

	torch::Tensor packed_text, packed_length;
	std::tie(packed_text, packed_length) = torch::_pack_padded_sequence(embedded, length.squeeze(1), false);

	auto rnn_out = rnn_->forward(packed_text, packed_length);

	auto hidden_state = rnn_out.state.narrow(0, 0, 1);
	hidden_state.squeeze(0);

	auto last_index = rnn_->options().num_layers() - 2;
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