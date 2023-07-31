#include "transformer.h"

AttentionHead::AttentionHead(int64_t embed_dim, int64_t head_dim) {
	q = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
	k = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
	v = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
}

at::Tensor AttentionHead::scaled_dot_product_attention(at::Tensor& query, at::Tensor& key, at::Tensor& value) {
	int64_t dim{query.size(-1)};
	at::Tensor scores{at::bmm(query, at::transpose(key, 1, 2)) / std::sqrt(dim)};
	at::Tensor weights{at::softmax(scores, -1)};
	return at::bmm(weights, value);
}


at::Tensor AttentionHead::forward(at::Tensor hidden_state) {
	at::Tensor qf = q(hidden_state);
	at::Tensor kf = k(hidden_state);
	at::Tensor vf = v(hidden_state);
	return scaled_dot_product_attention(qf, kf, vf);
}
