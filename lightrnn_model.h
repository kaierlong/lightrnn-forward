#ifndef _LIGHTRNN_MODEL_H_
#define _LIGHTRNN_MODEL_H_

#include <Eigen/Dense>
#include "predict_model.h"


class LightrnnModel: public PredictModel
{
private:
	Eigen::MatrixXf embed_r;
	Eigen::MatrixXf embed_c;
	
	std::vector<Eigen::MatrixXf> W;	
	std::vector<Eigen::VectorXf> b;	

	std::vector<Eigen::MatrixXf> c;
	std::vector<Eigen::MatrixXf> h;
	
	Eigen::MatrixXf softmax_W;		
	Eigen::VectorXf softmax_b;

	int vocab_size;
	int lightrnn_size;
	int embed_size;
	int num_hidden_layer;
	std::vector<int> layer_size;
	
	void computeRecurrentLayer(int layer, int r_id, int c_id);
	void computeLastState(int layer);
	void computeSimpleLastState(int layer, int r_id);
	void computeOutput(std::vector<std::pair<int,float>>& res, int K);
	void computeSimpleOutput(std::vector<std::pair<int,float>>& res, int K);

public:
	LightrnnModel(){};
	virtual bool init();
	virtual void predict(std::vector<std::vector<int>>& data, std::vector<std::vector<std::pair<int,float>>>& res, int K);
	virtual ~LightrnnModel(){};
};

#endif
