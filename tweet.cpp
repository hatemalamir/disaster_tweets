#include "tweet.h"

const std::regex Tweet::link{R"(https?://\S+)"};
const std::regex Tweet::hash{R"(#)"};
const std::regex Tweet::stock{R"(\$[_[:alnum:]]\S*)"};
const std::regex Tweet::rt{R"(^RT\s+)"};
const	std::regex Tweet::alnum{R"([\w|\s|']*)"};
const std::regex Tweet::token{R"([\s|,]+)"};

std::vector<std::string> &Tweet::stopwords() {
	static std::vector<std::string> st_ws;
	if(st_ws.empty()) {
		const std::string stopwords_file = "../data/stop_words_english.txt";
		std::ifstream ifs;
		ifs.open(stopwords_file, std::ios_base::in);
		if(!ifs) {
			throw std::invalid_argument("Failed to open stopwords file " + stopwords_file);
		}
		std::string ssw;
		ifs >> ssw;
		std::sregex_token_iterator swit{ssw.begin(), ssw.end(), token, -1};
		std::vector<std::string> st_ws_temp{swit, {}};
		st_ws = st_ws_temp;
	}
	return st_ws;
}

std::string Tweet::clean(const std::string text) {
	std::string clean_text = std::regex_replace(text, Tweet::link, "");
	clean_text = std::regex_replace(clean_text, hash, "");
	clean_text = std::regex_replace(clean_text, stock, "");
	clean_text = std::regex_replace(clean_text, rt, "");
	clean_text = regex_replace(clean_text, alnum, "$&", std::regex_constants::format_no_copy);
	for(size_t i = 0; i < clean_text.size(); i++)
		clean_text[i] = tolower(clean_text[i]);

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
				return s.size() == 0 || std::find(stopwords().begin(), stopwords().end(), s) != stopwords().end();
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
