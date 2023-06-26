#ifndef TWEET_H
#define TWEET_H

#include <string>
#include <vector>
#include <regex>
#include <iostream>

class Tweet {
	private:
		const static std::regex link;
		const static std::regex hash;
		const static std::regex stock;
		const static std::regex rt;
		const	static std::regex alnum;
		const static std::regex token;

		int id;
		std::string keyword;
		std::string location;
		std::string text;
		std::vector<std::string> clean_text;

		std::string clean(const std::string tweet);
		std::vector<std::string> tokenize(const std::string tweet);

	public:
		Tweet(int, std::string, std::string, std::string);
		Tweet(Tweet&& tweet);
		Tweet& operator=(Tweet&& tweet);

		size_t size() const {return clean_text.size();}
		const std::string& GetOriginal() const {return text;}
		const std::vector<std::string>& GetWords() const {return clean_text;}
};

#endif // TWEET_H
