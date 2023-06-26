#ifndef TWEETREADER_H
#define TWEETREADER_H

#include "tweet.h"
#include <string>
#include <vector>
#include <unordered_map>

class TweetReader {
private:
	using Tweets = std::vector<Tweet>;
	Tweets pos_samples_;
	Tweets neg_samples_;
	size_t pos_size_ = 0;
	size_t neg_size_ = 0;
	size_t max_size_ = 0;

	void ReadFile(const std::string& path, Tweets& pos_tweets, Tweets& neg_tweets);

public:
	TweetReader(const std::string& root_path);
	size_t GetPosSize() const {return pos_size_;}
	size_t GetNegSize() const {return neg_size_;}
	size_t GetMaxSize() const {return max_size_;}
	const Tweet& GetPos(size_t index) const {return pos_samples_[index];}
	const Tweet& GetNeg(size_t index) const {return neg_samples_[index];}
};

using WordsFrequencies = std::unordered_map<std::string, size_t>;
void GetWordsFrequencies(const TweetReader& reader, WordsFrequencies& frequencies);
void SelectTopFrequencies(WordsFrequencies& vocab, int64_t new_size);

#endif // TWEETREADER_H
