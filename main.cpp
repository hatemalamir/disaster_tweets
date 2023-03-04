#include "tweet_reader.h"

#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <utility>
#include <tuple>


int main(int argc, char** argv) {
	if(argc > 1) {
		auto root_path = std::filesystem::path(argv[1]);

		TweetReader train_reader(root_path);
		std::cout << "Read " << train_reader.GetPosSize() << " positive tweets, and " << train_reader.GetNegSize() << " negative tweets!" << std::endl;
		std::cout << "Max tweet size: " << train_reader.GetMaxSize() << std::endl;
	}
}
