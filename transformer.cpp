#include "transformer.h"

////////// AttentionHead //////////
AttentionHead::AttentionHead(int64_t embed_dim, int64_t head_dim) {
	q = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
	k = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
	v = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, head_dim).bias(true));
}

torch::Tensor AttentionHead::scaled_dot_product_attention(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value) {
	int64_t dim{query.size(-1)};
	torch::Tensor scores{at::bmm(query, at::transpose(key, 1, 2)) / std::sqrt(dim)};
	torch::Tensor weights{at::softmax(scores, -1)};
	return at::bmm(weights, value);
}


torch::Tensor AttentionHead::forward(torch::Tensor &hidden_state) {
	torch::Tensor qf = q(hidden_state);
	torch::Tensor kf = k(hidden_state);
	torch::Tensor vf = v(hidden_state);
	return scaled_dot_product_attention(qf, kf, vf);
}


////////// MultiHeadAttention //////////
MultiHeadAttention::MultiHeadAttention(int64_t embed_dim, int64_t num_heads) : embed_dim{embed_dim}, num_heads{num_heads}{
	int64_t head_dim = embed_dim / num_heads;

	heads = torch::nn::ModuleList();
	for(int64_t i = 0; i < num_heads; i++) {
		AttentionHead head{embed_dim, head_dim};
		heads->push_back(head);
	}

	output_linear = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(true));
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor &hidden_state) {
	std::vector<torch::Tensor> xs;
	for(const auto &head: *heads) {
		AttentionHead *h = head->as<AttentionHead>();
		xs.push_back(h->forward(hidden_state));
	}

	return output_linear(torch::cat(xs, -1));
}
