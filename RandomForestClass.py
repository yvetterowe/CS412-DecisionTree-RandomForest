import random
from DecisionTreeClass import *

class RandomForest(DecisionTree):
	
	def __init__(self, training_data, test_data, attribute_list, k, f):
		
		self._class_attribute_tuples = training_data
		self._class_attribute_tuples_test = test_data
		self._attribute_list = attribute_list
		self._tree_num = k		
		self._attr_num = f
		
		self._ensemble_trees = []
		self._processed_test_datasets = []
		
		random.seed()
	
	
	def generate_bootstrap_sample(self):
		sample_tuples = {}
		d = len(self._class_attribute_tuples)
		for i in range(d):
			random_tuple_id = random.randint(0, d-1)
			sample_tuples[i] = self._class_attribute_tuples[random_tuple_id]
		return sample_tuples
		
				
	def generate_ensemble_trees(self):
		for i in range(self._tree_num):
			d_i = self.generate_bootstrap_sample()
			test_data_copy = deepcopy(self._class_attribute_tuples_test)
			m_i = DecisionTree(d_i, test_data_copy, self._attribute_list, 100, 1) 
			m_i.construct(self._attr_num)
			self._ensemble_trees.append(m_i)
			
			
	def vote_for_class_label(self, tuple_info):
		votes = {-1:0, 1:0}
		for tree in self._ensemble_trees:
			curr_label = tree.predict_class_label(tuple_info)	
			votes[curr_label] += 1
		return -1 if votes[-1] > votes[1] else 1
	
	
	def run_dataset(self, class_attribute_tuples):
		output_result = {"TP":0, "FN":0, "FP":0, "TN":0}
		
		for tuple_id, tuple_info in class_attribute_tuples.items():
			real_label = tuple_info["class_label"]
			predict_label = self.vote_for_class_label(tuple_info)
			
			if predict_label == 1:
				if real_label == 1:
					output_result["TP"] += 1
				else:
					output_result["FP"] += 1
			else:
				if real_label == -1:
					output_result["TN"] += 1
				else:
					output_result["FN"] += 1
		
		return output_result
		
			
	def classify(self):
		self.generate_ensemble_trees()
		
		output_train = self.run_dataset(self._class_attribute_tuples)
		print output_train["TP"], output_train["FN"], output_train["FP"], output_train["TN"]
				
		output_test = self.run_dataset(self._class_attribute_tuples_test)
		print output_test["TP"], output_test["FN"], output_test["FP"], output_test["TN"]
		
		super(RandomForest, self).debug_evaluate(output_train)
		super(RandomForest, self).debug_evaluate(output_test)