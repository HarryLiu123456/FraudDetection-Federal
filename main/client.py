#clients.py用户模块

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import numpy
import time
import copy
import torch
import torch.utils.data

import utils

class Client(object):
	def __init__(self, conf, model, id = -1):
		self.conf = conf
		self.local_model = copy.deepcopy(model) 
		self.client_id = id
		self.optim = torch.optim.Adam(self.local_model.parameters(), 
                lr=self.conf["lr"], weight_decay=self.conf["weight_decay"])
  
	#这里传入的model是全局模型
	def local_train(self, model, loss, features, labels, train_g, test_g, test_mask,
				device, n_epochs, thresh, compute_metrics=False):
		#利用全局模型初始化本地模型
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
   
		#训练
		self.local_model.train()
		for epoch in range(n_epochs):
			tic = time.time()
			pred = self.local_model(train_g, features.to(device))
			l = loss(pred, labels)
			loss_val = l.data.item()

			self.optim.zero_grad()
			l.backward()
			self.optim.step()
			
			duration = time.time() - tic
			metric = utils.evaluate(self.local_model, train_g, features, labels, device)
			print("Id {:03d}, Local epoch {:03d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f} ".format(
					self.client_id, epoch, duration, loss_val, metric))	
		
  		#返回差异字典
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
		return diff

