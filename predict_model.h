#ifndef _PREDICT_MODEL_H_
#define _PREDICT_MODEL_H_

#include <string>
#include <vector>

class PredictModel
{
public:
	static PredictModel* createModel();
	static void destroyModel(PredictModel* pm);
	
	virtual bool init() = 0;	
	virtual void predict(std::vector<std::vector<int>>& data, std::vector<std::vector<std::pair<int, float>>>& res, int K) = 0;
	virtual ~PredictModel(){}	
};

#endif
