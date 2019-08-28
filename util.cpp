#include "util.h"
#include <math.h> 
#include <sstream>

void readVocab(std::string vocab_path, std::unordered_map<std::string,int>& vocab2id, std::vector<std::string>& id2vocab)
{
	std::ifstream vocab_file(vocab_path);
	std::string line;
	while(std::getline(vocab_file, line))	
	{
		std::istringstream iss(line);
		int id, cnt, class_id;
		std::string word;
		if (!(iss >> word))
		{
			printf("ERROR: vocab file wrong format! \n");
			exit(1);
		}
		id2vocab.push_back(word);
	}
	int vocab_size = id2vocab.size();	
	printf("vocab_size is %d\n", vocab_size);
	vocab_file.close();

	for(int i = 0; i < vocab_size; i++)
		vocab2id[id2vocab[i]] = i;
}

void readWord2rc(std::unordered_map<std::string, int>& vocab2id, std::unordered_map<std::string, std::pair<int, int>>& word2rc)
{
	int vocab_size = vocab2id.size();
	int lightrnn_size = sqrt(vocab_size);
	for(std::unordered_map<std::string, int>::iterator it = vocab2id.begin(); it != vocab2id.end(); ++it)
	{
		word2rc[it->first] = std::make_pair(it->second / lightrnn_size, it->second % lightrnn_size);
	}
}

std::vector<std::vector<int>> readBatchFromFile(std::istream& file, std::unordered_map<std::string, std::pair<int, int>>& word2rc, int max_input_len, int batch_size)
{
	std::string str;
	int sentence_num = 0;
	std::vector<std::vector<int>> data;
	while(std::getline(file, str))
	{
		std::istringstream ss(str);
		std::vector<std::string> full_sentence;
		std::string word;
		while(ss >> word)
			full_sentence.push_back(word);
		
		std::vector<std::string> inputs;
		if(full_sentence.size() > max_input_len)
		{
			inputs.assign(full_sentence.end()-max_input_len, full_sentence.end());
		} 
		else 
		{
			inputs.assign(full_sentence.begin(), full_sentence.end());
		}

		std::vector<int> rc_list;
		for(std::vector<std::string>::iterator it = inputs.begin(); it != inputs.end(); ++it)
		{
			std::string word = word2rc.find(*it) != word2rc.end() ? *it : "<unk>";
			rc_list.push_back(word2rc[word].first);
			rc_list.push_back(word2rc[word].second);
		}
		
		data.push_back(rc_list);
		sentence_num++;
		if(sentence_num == batch_size)
			break;
	}
	return data;
}

