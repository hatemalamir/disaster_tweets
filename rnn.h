#ifndef RNN_H
#define RNN_H

#include <torch/torch.h>

class SentimentRNNImpl : public torch::nn::Module {
	public:
		SentimentRNNImpl(
			int64_t vocab_size,
			int64_t embedding_dim,
			int64_t hidden_dim,
			int64_t output_dim,
			int64_t n_layers,
			bool bidirectional,
			double dropout,
			int64_t pad_idx
		);
		torch::Tensor forward(const torch::Tensor& text, const at::Tensor& length);
		void SetPretrainedEmbeddings(const at::Tensor& weights);

	private:
		int64_t pad_idx_{-1};
		torch::autograd::Variable embeddings_weights_;
		torch::nn::LSTM rnn_ = nullptr;
		torch::nn::Linear fc_ = nullptr;
		torch::nn::Dropout dropout_ = nullptr;
};
TORCH_MODULE(SentimentRNN);
#endif // RNN_H
