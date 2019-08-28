#include "util.h"
#include "predict_model.h"
#include <math.h>
#include <chrono>
#include <ctime>
#include <ratio>
#include <iostream>
#include <thread>

int main(int argc, char **argv)
{
		
	std::string vocab_path = "vocab_file";
	std::unordered_map<std::string,int> word2id;
	std::vector<std::string> id2word;
	readVocab(vocab_path, word2id, id2word);
	std::unordered_map<std::string, std::pair<int, int>> word2rc;
	readWord2rc(word2id, word2rc);
	int vocab_size = id2word.size();
	int lightrnn_size = sqrt(vocab_size);
	std::cout << "finish reading vocab_file" << std::endl;	
		
	// batch_size : # of sentences per batch
	int batch_size = 1;
	// num_step : # of words per query
	int max_input_len = 2;
		// topN : # of answers per query
	int topN = 3;
	
	// myPredictor would be nullptr if the following failed!
	auto myPredictor = PredictModel::createModel();
	if(myPredictor == nullptr)
		exit(1);
	std::cout << "finish loading model" << std::endl;	
		
	std::ifstream file("small_data_file");
	float total_time = 0;	
	int total_iter = 1;
	int iter = 0;
	while(iter < total_iter) 
	{
		//printf("iter: %d\n", iter);
		std::vector<std::vector<int>> data = readBatchFromFile(file, word2rc, max_input_len, batch_size);
		std::vector<std::vector<std::pair<int,float>>> res(data.size());
		
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();	
		myPredictor->predict(data, res, topN);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> time_span = t2 - t1;
		
		total_time += time_span.count();
		std::cout << "input: " << std::endl;
		for(int i = 0; i < data[0].size()/2; i++)
		{
			int id = data[0][2*i] * lightrnn_size + data[0][2*i+1];
			std::cout << id2word[id] << " ";
		}
		std::cout << std::endl;
		for(int i = 0; i < topN; i++)
		{
			std::cout << i << "th answer: " << id2word[res[0][i].first] << " with prob: " << res[0][i].second << std::endl;
		}
		iter++;
	}
	
	std::cout << "num sentence: " << total_iter << " average time per sentence: " << total_time/total_iter << std::endl;
	
	PredictModel::destroyModel(myPredictor);
	
	return 0;
}
