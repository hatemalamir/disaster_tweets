#include "tweet.h"

#include <regex>

const std::regex Tweet::re = std::regex("[^a-zA-Z0-9]");
const std::sregex_token_iterator Tweet::end;

Tweet::Tweet(int i, std::string k, std::string l, std::string t) :id{i}, keyword{k}, location{l}, text{t}{
	std::sregex_token_iterator token(text.begin(), text.end(), re, -1);
	for(; token != Tweet::end; ++token)
		if(token->length() > 1)
			clean_text.push_back(*token);
}

size_t Tweet::size() {
	return clean_text.size();
}
