#include "predict_model.h"
#include "lightrnn_model.h"

PredictModel* PredictModel::createModel()
{
	PredictModel* pm = new LightrnnModel();
	if(pm->init())
	{
		return pm;
	}
	destroyModel(pm);
	return nullptr;
}

void PredictModel::destroyModel(PredictModel* pm)
{
	delete pm;
}
