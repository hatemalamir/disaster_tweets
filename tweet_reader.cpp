#include "tweet_reader.h"
#include "csv.h"

#include <string>
#include <tuple>
#include <fstream>
#include <filesystem>
#include <future>
#include <utility>

TweetReader::TweetReader(const std::string& root_path) {
	auto file_path = std::filesystem::path(root_path);
	if(std::filesystem::exists(file_path)) {
		auto samples = std::async(std::launch::async, [&](){ReadFile(file_path, pos_samples_, neg_samples_);});
		samples.get();
	}
	else {
		throw std::invalid_argument(">>> TweetReader: Incorrect path");
	}
}

template <std::size_t... Idx, typename T, typename R>
bool  read_row_helper(std::index_sequence<Idx...>, T& row, R& reader) {
	return  reader.read_row(std::get<Idx>(row)...);
}

template <std::size_t... Idx, typename R>
std::unique_ptr<Tweet> fill_values(std::index_sequence<Idx...>, R& row) {
	std::unique_ptr<Tweet> tweet{new Tweet(std::get<Idx>(row)...)};
	return std::move(tweet);
}

void TweetReader::ReadFile(const std::string& path, Tweets& pos_tweets, Tweets& neg_tweets) {
	std::ifstream file(path);
	if(file) {
		const uint32_t columns_num = 5;
		io::CSVReader<columns_num, io::trim_chars<' '>, io::double_quote_escape<',','\"'>> csv_reader(path);

		using RowType = std::tuple<int, std::string, std::string, std::string, int>;
		RowType row;
		try {
			csv_reader.read_header(io::ignore_extra_column, "id", "keyword", "location", "text", "target");
			bool done = false;
			while(!done) {
				done = !read_row_helper(std::make_index_sequence<std::tuple_size<RowType>::value>{}, row, csv_reader);
				if(!done) {
					int target = std::get<4>(row);
					std::unique_ptr<Tweet> tweet = fill_values(std::make_index_sequence<columns_num - 1>{}, row);
					max_size_ = std::max(max_size_, tweet->size());
					if(target == 1)
						pos_tweets.push_back(std::move(tweet));
					else if(target == 0)
						neg_tweets.push_back(std::move(tweet));
					else
						throw std::invalid_argument(">>> TweetReader: Invalid target value");
				}
			}
		} catch(io::error::no_digit& err) {
			std::cerr << err.what() << std::endl;
		}
	}
}

void GetWordsFrequencies(const TweetReader& reader, WordsFrequencies& frequencies) {
	for(size_t i = 0; i < reader.GetPosSize(); ++i) {
		const Tweet& tweet = reader.GetPos(i);
		for(const std::string& word : tweet.GetWords()) {
			frequencies[word] += i;
		}
	}
	for(size_t i = 0; i < reader.GetNegSize(); ++i) {
		const Tweet& tweet = reader.GetNeg(i);
		for(const std::string& word : tweet.GetWords()) {
			frequencies[word] += i;
		}
	}
}

void SelectTopFrequencies(WordsFrequencies& vocab, int64_t new_size) {
	using FreqItem = std::pair<size_t, WordsFrequencies::iterator>;
	std::vector<FreqItem> freq_items;
	freq_items.reserve(vocab.size());
	auto i = vocab.begin();
	for(; i != vocab.end(); ++i)
		freq_items.push_back({i->second, i});

	std::sort(
			freq_items.begin(),
			freq_items.end(),
			[](const FreqItem& a, const FreqItem& b) {return a.first < b.first;});

	std::reverse(freq_items.begin(), freq_items.end());

	freq_items.resize(static_cast<size_t>(new_size));

	WordsFrequencies new_vocab;
	int pi = 0;
	std::cout << ">>> Top 20 words:" << std::endl;
	for(auto& item: freq_items) {
		new_vocab.insert({item.second->first, item.first});
		if(pi < 20)
			std::cout << "  " << item.second->first << ": " << item.first << std::endl;
		pi++;
	}
	vocab = new_vocab;
}
