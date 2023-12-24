
import torch

class Server(object): 
	def __init__(self, conf, model):
		self.conf = conf
		self.global_model = model

	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)