#include "glove_reader.h"
#include "tweet_dataset.h"
#include "tweet_reader.h"
#include "rnn.h"
#include "vocabulary.h"

#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;


float BinaryAccuracy(const torch::Tensor& preds, const torch::Tensor& target) {
	auto rounded_preds = torch::round(torch::sigmoid(preds));
	
	auto correct = torch::eq(rounded_preds, target).to(torch::dtype(torch::kFloat));

	auto acc = correct.sum() / correct.size(0);

	return acc.item<float>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MakeBatchTensors(const std::vector<TweetSample>& batch) {
	std::vector<torch::Tensor> text_data;
	std::vector<torch::Tensor> text_lengths;
	std::vector<torch::Tensor> label_data;
	for(auto item : batch) {
		text_data.push_back(item.data.first);
		text_lengths.push_back(item.data.second);
		label_data.push_back(item.target);
	}

	std::vector<size_t> permutation(text_lengths.size());
	std::iota(permutation.begin(), permutation.end(), 0);
	std::sort(
		permutation.begin(),
		permutation.end(),
		[&](std::size_t i, std::size_t j) {
			return text_lengths[i].item().toLong() < text_lengths[j].item().toLong();
		}
	);
	std::reverse(permutation.begin(), permutation.end());

	auto apply_permutation = [&permutation] (const std::vector<torch::Tensor>& vec) {
		std::vector<torch::Tensor> sorted_vec(vec.size());
		std::transform(
			permutation.begin(),
			permutation.end(),
			sorted_vec.begin(),
			[&](std::size_t i) {
			  return vec[i];
			}
		);
		return sorted_vec;
	};

	text_data = apply_permutation(text_data);
	text_lengths = apply_permutation(text_lengths);
	label_data = apply_permutation(label_data);

	torch::Tensor texts = torch::stack(text_data);
	torch::Tensor lengths = torch::stack(text_lengths);
	torch::Tensor labels = torch::stack(label_data);

	return {texts, lengths, labels};
}

void TrainModel(
	int epoch,
	SentimentRNN& model,
	torch::optim::Optimizer& optimizer,
	torch::data::StatelessDataLoader<TweetDataset, torch::data::samplers::RandomSampler>& train_loader
) {
	model->train();

	double epoch_loss = 0;
	double epoch_acc = 0;

	int batch_index = 0;
	for(auto& batch: train_loader) {
		optimizer.zero_grad();

		torch::Tensor texts, lengths, labels;
		std::tie(texts, lengths, labels) = MakeBatchTensors(batch);

		torch::Tensor predictions = model->forward(texts.t(), lengths);
		predictions.squeeze_(1);

		torch::Tensor loss = torch::binary_cross_entropy_with_logits(
			predictions,
			labels,
			{},
			{},
			at::Reduction::Mean
		);

		loss.backward();

		optimizer.step();

		auto loss_value = static_cast<double>(loss.item<float>());
		auto acc_value = static_cast<double>(BinaryAccuracy(predictions, labels));

		epoch_loss += loss_value;
		epoch_acc += acc_value;

		if(++batch_index % 10 == 0)
			std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss_value << " Accuracy: " << acc_value << std::endl;
	}

	std::cout << "Epoch: " << epoch << " | Loss: " << (epoch_loss / (batch_index - 1))  << " Accuracy: " << (epoch_acc / (batch_index - 1)) << std::endl;
}

int main(int argc, char** argv) {
	if(argc > 1) {
		torch::DeviceType device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;

		auto train_path = fs::path(argv[1]);

		TweetReader train_reader(train_path);
		std::cout << ">>> Training dataset loaded! Read " << train_reader.GetPosSize() << " positive tweets, and " << train_reader.GetNegSize() << " negative tweets." << " Max tweet size: " << train_reader.GetMaxSize() << std::endl;

		WordsFrequencies words_frequencies;
		GetWordsFrequencies(train_reader, words_frequencies);
		int64_t vocab_size = 25000;
		std::cout << ">>> Keeping the " << vocab_size << " most frequent words" << std::endl;
		SelectTopFrequencies(words_frequencies, vocab_size);

		int64_t embedding_dim = 200;
		GloveReader glove_reader(argv[2], embedding_dim);
		Vocabulary vocab(words_frequencies, glove_reader);
		std::cout << ">>> Added " << vocab.GetEmbeddingsCount() << " words with embeddings to the vocabulary" << std::endl;

		TweetDataset train_dataset(&train_reader, &vocab, device);
		size_t batch_size = 32;
		auto train_loader = torch::data::make_data_loader(
			train_dataset,
			torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
		);
		/*
		auto test_loader = torch::data::make_data_loader(
			test_dataset,
			torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
		);
		*/

		int64_t hidden_dim = 256;
		int64_t output_dim = 1;
		int64_t n_layers = 2;
		bool bidirectional = true;
		double dropout = 0.5;
		int64_t pad_idx = vocab.GetPaddingIndex();
		SentimentRNN model(
			vocab.GetEmbeddingsCount(),
			embedding_dim,
			hidden_dim,
			output_dim,
			n_layers,
			bidirectional,
			dropout,
			pad_idx
		);
		model->SetPretrainedEmbeddings(vocab.GetEmbeddings());

		double learning_rate = 0.01;
		torch::optim::Adam optimizer(
			model->parameters(),
			torch::optim::AdamOptions(learning_rate)
		);

		model->to(device);

		int epochs = 100;
		for(int epoch = 0; epoch < epochs; ++epoch) {
			TrainModel(epoch, model, optimizer, *train_loader);
			// TestModel();
		}

		return 0;
	}
	else 
		std::cerr << "Please specify a path to the dataset and glove vector file\n";
	return 1;
}
