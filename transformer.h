#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>

class AttentionHead : public torch::nn::Module {
	private:
		torch::nn::Linear q = nullptr;
		torch::nn::Linear k = nullptr;
		torch::nn::Linear v = nullptr;

		at::Tensor scaled_dot_product_attention(at::Tensor& query, at::Tensor& key, at::Tensor& value);
	public:
		AttentionHead(int64_t embed_dim, int64_t head_dim);
		at::Tensor forward(at::Tensor hidden_state);
};

class MultiHeadAttention : public torch::nn::Module {
};

#endif //TRANSFORMER.h
