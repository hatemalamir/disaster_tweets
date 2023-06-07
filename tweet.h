#ifndef TWEET_H
#define TWEET_H

#include <string>
#include <vector>
#include <regex>
#include <iostream>

class Tweet {
	private:
		int id;
		std::string keyword;
		std::string location;
		std::string text;
		std::vector<std::string> clean_text;

		const static std::regex re;
		const static std::sregex_token_iterator end;

	public:
		Tweet(int, std::string, std::string, std::string);

		size_t size() const {return clean_text.size();}
		const std::vector<std::string>& GetWords() const {return clean_text;}
		// ~Tweet(){std::cout << "Tweet id " << id << " deleted";}
};

#endif // TWEET_H
