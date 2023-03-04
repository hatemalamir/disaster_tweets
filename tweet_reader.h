#include "tweet.h"
#include <string>
#include <vector>

class TweetReader {
	private:
		using Tweets = std::vector<std::unique_ptr<Tweet> >;

		Tweets pos_samples_;
		Tweets neg_samples_;
		Tweets test_samples_;

		size_t max_size_ = 0;

		void ReadTrainFile(const std::string& path, Tweets& pos_tweets, Tweets& neg_tweets);
		void ReadTestFile(const std::string& path, Tweets& test_tweets);

	public:
		TweetReader(const std::string& root_path);
		size_t GetPosSize() const {return pos_samples_.size();}
		size_t GetNegSize() const {return neg_samples_.size();}
		size_t GetMaxSize() const {return max_size_;}
		size_t GetTestSize() const;
		const Tweet& GetPos(size_t index) const;
		const Tweet& GetNeg(size_t index) const;
		const Tweet& GetTest(size_t index) const;
};
