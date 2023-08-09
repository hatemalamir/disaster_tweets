#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>

class AttentionHead : public torch::nn::Module {
	private:
		torch::nn::Linear q = nullptr;
		torch::nn::Linear k = nullptr;
		torch::nn::Linear v = nullptr;

		torch::Tensor scaled_dot_product_attention(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value);
	public:
		AttentionHead(int64_t embed_dim, int64_t head_dim);
		torch::Tensor forward(torch::Tensor &hidden_state);
};

class MultiHeadAttention : public torch::nn::Module {
	private:
		int64_t embed_dim;
		int64_t num_heads;
		torch::nn::ModuleList heads = nullptr;
		torch::nn::Linear output_linear = nullptr;
	public:
		MultiHeadAttention(int64_t embed_dim, int64_t num_heads);
		torch::Tensor forward(torch::Tensor &hidden_state);
};

#endif //TRANSFORMER.h
