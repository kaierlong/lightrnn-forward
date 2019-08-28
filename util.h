#ifndef _UTIL_H_
#define _UTIL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <utility> 
#include <fstream>

void readVocab(std::string vocab_path, std::unordered_map<std::string, int>& vocab2id, std::vector<std::string>& id2vocab);

void readWord2rc(std::unordered_map<std::string, int>& vocab2id, std::unordered_map<std::string, std::pair<int, int>>& word2rc);

std::vector<std::vector<int>> readBatchFromFile(std::istream& file, std::unordered_map<std::string, std::pair<int, int>>& word2rc, int max_input_len, int batch_size);

#endif

