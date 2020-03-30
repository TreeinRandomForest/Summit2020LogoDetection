from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader

def evaluate(dataset_name, model_output_dir, cfg, model):
	evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=model_output_dir)
	loader = build_detection_test_loader(cfg, dataset_name)
	metrics = inference_on_dataset(model, loader, evaluator)

	return metrics