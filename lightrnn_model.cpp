#include <iostream>
#include <stdexcept>
#include <algorithm> 
#include <float.h>
#include <math.h>
#include "lightrnn_model.h"

using namespace Eigen;

float my_sigmoid(float a)
{
	return 1/(1+exp(-a));
}

float my_tanh(float a)
{
	return tanhf(a);
}

VectorXf softmax(VectorXf& vec)
{
	float max = vec.maxCoeff();
	vec = (vec.array() - max).exp();
	float sum = vec.sum();
	VectorXf res = vec.array() / sum;
	return res;
}

MatrixXf softmax(MatrixXf& mat)
{
	RowVectorXf col_max = mat.colwise().maxCoeff();
	MatrixXf res = (mat.array().rowwise() - col_max.array()).exp();
	RowVectorXf col_sum = res.colwise().sum();
	res = res.array().rowwise() / col_sum.array();
	return res;
}

void topK(VectorXf vec, std::vector<std::pair<int, float>>& res, int K)
{
	for(int i = 0; i < K; i++)
	{
		VectorXf::Index max_ind;
		float max_score = vec.maxCoeff(&max_ind);
		vec(max_ind) = -FLT_MAX;
		std::pair<int, float> p = std::make_pair(max_ind, max_score);
		res.push_back(p);
	}
}

void goToDelimiter(int delim, FILE *fi)
{
	int ch=0;
	while (ch != delim) {
		ch = fgetc(fi);
		if (feof(fi)) {
			std::cout << "Unexpected end of file" << std::endl;
			throw std::exception();
		}
	}
}

bool loadWeight(const char *file_name, MatrixXf& mat, int row, int col)
{
	FILE *pFile = fopen(file_name, "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}

	float fl;
	mat.resize(row, col);
	for(int r = 0; r < row; r++)
	{
		for(int c = 0; c < col; c++)
		{
			fread(&fl, 4, 1, pFile);
			if (std::isnan(fl))
			{
				printf("result is nan \n");
				return false;
			}
			mat(r, c) = fl;
		}
	}
	mat.transposeInPlace();
	fclose(pFile);
	return true;
}

bool loadWeight(const char *file_name, VectorXf& vec, int size)
{
	FILE *pFile = fopen(file_name, "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}

	float fl;
	vec.resize(size);
	for(int i = 0; i < size; i++)
	{
		fread(&fl, 4, 1, pFile);
		if (std::isnan(fl))
		{
			printf("result is nan \n");
			return false;
		}
		vec(i) = fl;			
	}
	fclose(pFile);
	return true;
}

bool LightrnnModel::init()
{	
	FILE *pFile = fopen("model_config", "rb");
	if(pFile == NULL) 
	{
		std::cout << "Model file is not found!" << std::endl;
		return false;
	}
		
	try 
	{	
		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &vocab_size);

		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &lightrnn_size);

		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &embed_size);

		goToDelimiter(':', pFile);
		fscanf(pFile, "%d", &num_hidden_layer);
			
		int layersize;
		for(int layer = 0; layer < num_hidden_layer; layer++)
		{
			goToDelimiter(':', pFile);
			fscanf(pFile, "%d", &layersize);
			layer_size.push_back(layersize);
		}
	} catch(const std::exception& e) {
		return false;
	}
	fclose(pFile);
	
	loadWeight("embedding_r", embed_r, lightrnn_size, embed_size);	
	loadWeight("embedding_c", embed_c, lightrnn_size, embed_size);	
	
	W = std::vector<MatrixXf>(num_hidden_layer);
	b = std::vector<VectorXf>(num_hidden_layer);
	c = std::vector<MatrixXf>(num_hidden_layer);
	h = std::vector<MatrixXf>(num_hidden_layer);
	
	for(int layer = 0; layer < num_hidden_layer; layer++)
	{
		int	nRows = layer_size[layer] + (layer == 0 ? embed_size : layer_size[layer-1]);
		int nCols= layer_size[layer];
	
		loadWeight("kernel", W[layer], nRows, 4*nCols);	
		loadWeight("bias", b[layer], 4*nCols);	
	}
	
	loadWeight("softmax_w", softmax_W, layer_size[num_hidden_layer-1], lightrnn_size);	
	loadWeight("softmax_b", softmax_b, lightrnn_size);	
	
	return true;
}

void LightrnnModel::computeRecurrentLayer(int layer, int r_id, int c_id)
{
	if(layer == 0)
	{
		int hidden_size = layer_size[layer];
		VectorXf input_r = embed_r.col(r_id);
		VectorXf concats = W[layer] * (VectorXf(embed_size + hidden_size) << input_r, h[layer]).finished()  + b[layer];
		VectorXf i = concats.segment(0, hidden_size);
		VectorXf j = concats.segment(hidden_size, hidden_size);
		VectorXf f = concats.segment(hidden_size*2, hidden_size);
		VectorXf o = concats.segment(hidden_size*3, hidden_size);
	
		c[layer] = c[layer].array() * f.unaryExpr(std::ptr_fun(my_sigmoid)).array() + i.unaryExpr(std::ptr_fun(my_sigmoid)).array() * j.unaryExpr(std::ptr_fun(my_tanh)).array();
		h[layer] = c[layer].unaryExpr(std::ptr_fun(my_tanh)).array() * o.unaryExpr(std::ptr_fun(my_sigmoid)).array();
		
		VectorXf input_c = embed_c.col(c_id);
		concats = W[layer] * (VectorXf(embed_size + hidden_size) << input_c, h[layer]).finished() + b[layer];
		i = concats.segment(0, hidden_size);
		j = concats.segment(hidden_size, hidden_size);
		f = concats.segment(hidden_size*2, hidden_size);
		o = concats.segment(hidden_size*3, hidden_size);
		
		c[layer] = c[layer].array() * f.unaryExpr(std::ptr_fun(my_sigmoid)).array() + i.unaryExpr(std::ptr_fun(my_sigmoid)).array() * j.unaryExpr(std::ptr_fun(my_tanh)).array();
		h[layer] = c[layer].unaryExpr(std::ptr_fun(my_tanh)).array() * o.unaryExpr(std::ptr_fun(my_sigmoid)).array();
	}
	else
	{
		// This model has only one layer
	}	
}

void LightrnnModel::computeLastState(int layer)
{
	if(layer == 0)
	{
		int hidden_size = layer_size[layer];
		MatrixXf input_rs = embed_r;
		MatrixXf hs(hidden_size, lightrnn_size);
		MatrixXf cs(hidden_size, lightrnn_size);
		for(int i = 0; i < lightrnn_size; i++)
		{
			hs.col(i) = h[layer];
			cs.col(i) = c[layer];
		}
		MatrixXf concats = (W[layer] * (MatrixXf(embed_size + hidden_size, lightrnn_size) << input_rs, hs).finished()).colwise() + b[layer];
		MatrixXf i = concats.block(0, 0, hidden_size, lightrnn_size);
		MatrixXf j = concats.block(hidden_size, 0, hidden_size, lightrnn_size);
		MatrixXf f = concats.block(hidden_size*2, 0, hidden_size, lightrnn_size);
		MatrixXf o = concats.block(hidden_size*3, 0, hidden_size, lightrnn_size);
		
		c[layer] = cs.array() * f.unaryExpr(std::ptr_fun(my_sigmoid)).array() + i.unaryExpr(std::ptr_fun(my_sigmoid)).array() * j.unaryExpr(std::ptr_fun(my_tanh)).array();
		h[layer] = c[layer].unaryExpr(std::ptr_fun(my_tanh)).array() * o.unaryExpr(std::ptr_fun(my_sigmoid)).array();
	}
	else
	{
		// This model has only one layer
	}	
}

void LightrnnModel::computeSimpleLastState(int layer, int r_id)
{
	if(layer == 0)
	{
		int hidden_size = layer_size[layer];
		VectorXf input_r = embed_r.col(r_id);
		VectorXf concats = W[layer] * (VectorXf(embed_size + hidden_size) << input_r, h[layer]).finished()  + b[layer];
		VectorXf i = concats.segment(0, hidden_size);
		VectorXf j = concats.segment(hidden_size, hidden_size);
		VectorXf f = concats.segment(hidden_size*2, hidden_size);
		VectorXf o = concats.segment(hidden_size*3, hidden_size);
	
		c[layer] = c[layer].array() * f.unaryExpr(std::ptr_fun(my_sigmoid)).array() + i.unaryExpr(std::ptr_fun(my_sigmoid)).array() * j.unaryExpr(std::ptr_fun(my_tanh)).array();
		h[layer] = c[layer].unaryExpr(std::ptr_fun(my_tanh)).array() * o.unaryExpr(std::ptr_fun(my_sigmoid)).array();
	}
	else
	{
		// This model has only one layer
	}	
}

void LightrnnModel::computeOutput(std::vector<std::pair<int,float>>& res, int K)
{
	VectorXf logit_r = softmax_W * h[num_hidden_layer-1] + softmax_b;
	VectorXf prob_r = softmax(logit_r);
	for(int layer = 0; layer < num_hidden_layer; layer++)
	{
		computeLastState(layer);
	}
	MatrixXf logit_c = (softmax_W * h[num_hidden_layer-1]).colwise() + softmax_b;
	MatrixXf prob_c = softmax(logit_c);
	VectorXf prob(vocab_size);	
	for(int r_id = 0; r_id < lightrnn_size; r_id++)
	{
		for(int c_id = 0; c_id < lightrnn_size; c_id++)
		{
			prob(r_id*lightrnn_size + c_id) = prob_r(r_id) * prob_c(c_id, r_id);
		}
	}
	topK(prob, res, K);
}

void LightrnnModel::computeSimpleOutput(std::vector<std::pair<int,float>>& res, int K)
{
	VectorXf logit_r = softmax_W * h[num_hidden_layer-1] + softmax_b;
	VectorXf prob_r = softmax(logit_r);
	VectorXf::Index max_r_id;
	float max_prob_r = prob_r.maxCoeff(&max_r_id);
	for(int layer = 0; layer < num_hidden_layer; layer++)
	{
		computeSimpleLastState(layer, max_r_id);
	}
	VectorXf logit_c = softmax_W * h[num_hidden_layer-1] + softmax_b;
	VectorXf prob_c = softmax(logit_c);
	VectorXf prob(vocab_size);	
	for(int r_id = 0; r_id < lightrnn_size; r_id++)
	{
		for(int c_id = 0; c_id < lightrnn_size; c_id++)
		{
			prob(r_id*lightrnn_size + c_id) = prob_r(r_id) * prob_c(c_id);
		}
	}
	topK(prob, res, K);
}

void LightrnnModel::predict(std::vector<std::vector<int>>& data, std::vector<std::vector<std::pair<int, float>>>& res, int K)
{	
	for(int i = 0; i < data.size(); i++)
	{
		for(int layer = 0; layer < num_hidden_layer; layer++)
		{
			h[layer].resize(layer_size[layer], 1);
			h[layer].setZero();
			c[layer].resize(layer_size[layer], 1);
			c[layer].setZero();
		}
		
		for(int step = 0; step < data[i].size()/2; step++)
		{
			for(int layer = 0; layer < num_hidden_layer; layer++)
			{
				computeRecurrentLayer(layer, data[i][step*2], data[i][step*2+1]);
			}
		}
		//std::cout << h[num_hidden_layer-1] << std::endl;
		//computeOutput(res[i], K);
		computeSimpleOutput(res[i], K);
	}
}


		
