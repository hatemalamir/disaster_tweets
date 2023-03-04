#include "tweet_reader.h"
#include "csv.h"

#include <string>
#include <tuple>
#include <fstream>
#include <filesystem>
#include <future>
#include <utility>

TweetReader::TweetReader(const std::string& root_path) {
	auto root = std::filesystem::path(root_path);
	auto train_path = root / "train.csv";
	auto test_path = root / "test.csv";
	if(std::filesystem::exists(train_path) && std::filesystem::exists(test_path)) {
		auto training = std::async(std::launch::async, [&](){ReadTrainFile(train_path, pos_samples_, neg_samples_);});
		// auto test = std::async(std::launch::async, [&](){ReadTestFile(test_path, test_samples_);});
		training.get();
	}
	else {
		throw std::invalid_argument("TweetReader: Incorrect path");
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

void TweetReader::ReadTrainFile(const std::string& path, Tweets& pos_tweets, Tweets& neg_tweets) {
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
						throw std::invalid_argument("TweetReader: Invalid target value");
				}
			}
			std::cout << ">>> Loaded " << pos_tweets.size() << " positive tweets, and " << neg_tweets.size() << " negative tweets" << std::endl;
		} catch(io::error::no_digit& err) {
			std::cerr << err.what() << std::endl;
		}
	}
}
