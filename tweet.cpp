#include "tweet.h"

const std::regex Tweet::link{R"(https?://\S+)"};
const std::regex Tweet::hash{R"(#)"};
const std::regex Tweet::stock{R"(\$[_[:alnum:]]\S*)"};
const std::regex Tweet::rt{R"(^RT\s+)"};
const	std::regex Tweet::alnum{R"([\w|\s]*)"};
const std::regex Tweet::token{R"([\s|,]+)"};

std::string Tweet::clean(const std::string text) {
	std::string clean_text = std::regex_replace(text, Tweet::link, "");
	clean_text = std::regex_replace(clean_text, hash, "");
	clean_text = std::regex_replace(clean_text, stock, "");
	clean_text = std::regex_replace(clean_text, rt, "");
	clean_text = regex_replace(clean_text, alnum, "$&", std::regex_constants::format_no_copy);

	return clean_text;
}

std::vector<std::string> Tweet::tokenize(const std::string text) {
	std::sregex_token_iterator it{text.begin(), text.end(), token, -1};
	std::vector<std::string> tokenized{it, {}};

	tokenized.erase(
		std::remove_if(
			tokenized.begin(),
			tokenized.end(),
			[](std::string const& s){
				return s.size() == 0;
			}),
		tokenized.end()
	);

	return tokenized;
}

Tweet::Tweet(int i, std::string k, std::string l, std::string t) :id{i}, keyword{k}, location{l}, text{t}{
	std::string clean_tweet = clean(text);
	clean_text = tokenize(clean_tweet);
}

Tweet::Tweet(Tweet&& tweet) : id(tweet.id), keyword(std::move(tweet.keyword)), location(std::move(tweet.location)), text(std::move(tweet.text)), clean_text(std::move(tweet.clean_text)) {
	tweet.id = -1;
}

Tweet& Tweet::operator=(Tweet&& tweet) {
	id = tweet.id;
	tweet.id = -1;
	keyword = std::move(tweet.keyword);
	location = std::move(tweet.location);
	text = std::move(tweet.text);
	clean_text = std::move(tweet.clean_text);

	return *this;
}
