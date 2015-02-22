import sys
import re
import math
import operator
import random
from copy import deepcopy

class DecisionTree(object):
	
	def __init__(self, training_data, test_data, attribute_list, max_depth, min_split_size):
		
		self._class_attribute_tuples= training_data
		self._class_attribute_tuples_test = test_data
		self._attribute_list = attribute_list
		self._max_depth = max_depth #must be an even number for now
		self._min_split_size = min_split_size
		
		self._attribute_values_dict = {}
		for attr_index in attribute_list:
			self._attribute_values_dict[attr_index] = set()
	
		self._class_attribute_values_count = {}
		self._class_attribute_values_count_test = {}

		self._tree = {}
		self._curr_max_depth = -1;
		
		self._output_result = {"TP":0, "FN":0, "FP":0, "TN":0}
		self._output_result_test = {"TP":0, "FN":0, "FP":0, "TN":0}
		
		random.seed()
	
	
	
	'''
		preprocess data methods
	'''
	def build_attribute_value_dict(self):
		for tuple_id, tuple_info in self._class_attribute_tuples.items():
			for attr_index in self._attribute_list:
				self._attribute_values_dict[attr_index].update([tuple_info[attr_index]])
		
		#print self._attribute_values_dict
	
	
	def build_class_attribute_value_count_dict(self, class_attribute_tuples):
		class_attribute_values_count = {-1:{}, 1:{}}
		negative_count = class_attribute_values_count[-1]
		positive_count = class_attribute_values_count[1]
		
		for attr_index in self._attribute_values_dict:
			negative_count[attr_index] = {}
			positive_count[attr_index] = {}
			attr_values = self._attribute_values_dict[attr_index]
			for attr_value in attr_values:
				negative_count[attr_index][attr_value] = 0
				positive_count[attr_index][attr_value] = 0
		
		tuples_copy = deepcopy(class_attribute_tuples)
		for tuple_id, tuple_info in tuples_copy.items():
			curr_count = class_attribute_values_count[tuple_info["class_label"]]
			for attr_index in tuple_info:
				if attr_index == "class_label":
					continue
					
				if not attr_index in self._attribute_list:
					del class_attribute_tuples[tuple_id][attr_index]
					continue
					
				attr_value = tuple_info[attr_index]
				if not attr_value in self._attribute_values_dict[attr_index]:
					#print "aha got you!"
					#print attr_index, attr_value
					#mark it as -1 for now and replace it with "most common value" later
					class_attribute_tuples[tuple_id][attr_index] = -1
					continue
				
				curr_count[attr_index][attr_value] += 1
				
		return class_attribute_values_count
		
				
	def replace_unexpected_value_with_common_value(self, class_attribute_tuples, class_attribute_values_count):
		for tuple_id, tuple_info in class_attribute_tuples.items():
			class_label = tuple_info["class_label"]
			for attr_index in self._attribute_list:
				if tuple_info[attr_index] == -1:
					curr_value_count = class_attribute_values_count[class_label][attr_index]
					tuple_info[attr_index] = max(curr_value_count.iteritems(), key=operator.itemgetter(1))[0]
	
					
	def pre_process_data(self):
		self.build_attribute_value_dict()										
		self._class_attribute_values_count_test = self.build_class_attribute_value_count_dict(self._class_attribute_tuples_test)
		self.replace_unexpected_value_with_common_value(self._class_attribute_tuples_test, self._class_attribute_values_count_test)	
		
		
	def build_avc_group(self, data, attribute_list):
		avc_group = {}		
		
		#initialize avc group
		for attr_index, attr_values in attribute_list.items():
			avc_group[attr_index] = {}
			for attr_value in attr_values:
				avc_group[attr_index][attr_value] = {-1:0, 1:0}
		
		#build avc set for each attribute 
		for tuple in data.values():
			class_label = tuple["class_label"]
			for attr_index in attribute_list.keys():
				attr_value = tuple[attr_index]
				avc_group[attr_index][attr_value][class_label] += 1

		return avc_group
	'''									
		*********************************************************************************
	'''
	
	
	
	'''
		splitting criterion computing methods
	'''	
	def entropy(self, data_list):
		result = 0.0	
		total = sum(data_list)
		for var in data_list:
			if var:
				result += (-float(var) / total) * math.log(float(var) / total, 2)
		return result
	
	
	def information_entropy(self, data):
		class_num = {-1:0, 1:0}
		for tuple in data.values():
			class_label = tuple["class_label"]
			class_num[class_label] += 1
		return self.entropy(class_num.values())
		
	
	def information_need(self, avc_set, tuple_num):
		result = 0.0
		for class_count in avc_set.values():
			count = class_count.values()
			prob = float(sum(count)) / tuple_num
			result += prob * self.entropy(count)
		return result
		
		
	def information_gain(self, avc_set, data_entropy, tuple_num):
		return data_entropy  - self.information_need(avc_set, tuple_num)
							
	
	def split_information(self, avc_set, tuple_num):
		value_count_list = []
		for value_class_count in avc_set.values():
			value_count_list.append(sum(value_class_count.values()))
		return self.entropy(value_count_list)
	
		
	def gain_ratio(self, info_gain, avc_set,tuple_num):
		return info_gain / self.split_information(avc_set, tuple_num)
	
	
	def gini_index(self, data_list):
		result = 1.0
		total = sum(data_list)
		for var in data_list:
			if var:
				result -= (-float(var) / total) ** 2
		return result
			
			
	def gini_index_attr(self, avc_set, tuple_num):
		result = 0.0
		for class_count in avc_set.values():
			count = class_count.values()
			prob = float(sum(count)) / tuple_num
			result += prob * self.gini_index(count)
		return result

		
	def reduction_in_impurity(self, avc_set, data_gini, tuple_num):
		return data_gini - self.gini_index_attr(avc_set, tuple_num)
			
			
	#choose smallest gini index on split attribute
	def choose_best_attribute_cart(self, data, attribute_list):
		best_attr = 0
		min_measure = 1.0
		curr_avc_group = self.build_avc_group(data, attribute_list)
		for attr in attribute_list:			
			split_attr_gini_index = self.gini_index_attr(curr_avc_group[attr], len(data))		
			if split_attr_gini_index < min_measure:
				min_measure = split_attr_gini_index
				best_attr = attr				
		return best_attr
			
				
	def choose_best_attribute_gain_raio(self, data, attribute_list):
		best_attr = 0
		max_measure = -1.0
		curr_avc_group = self.build_avc_group(data, attribute_list)
		data_entropy = self.information_entropy(data)
		
		#compute average information gain 
		#use it as the constraint threshold to avoid unstable gain ratio (information_split -> 0)
		attr_info_gain = {}
		for attr in attribute_list:
			attr_info_gain[attr] = self.information_gain(curr_avc_group[attr], data_entropy, len(data))
		avg_info_gain = sum(attr_info_gain.values()) / len(attr_info_gain.values())
		
		#if really pretttttttty small, choose randomly
		if avg_info_gain == 0.0:
			return random.sample(attribute_list.keys(),1)[0]
		
		for attr in attribute_list:
			if attr_info_gain[attr] < avg_info_gain:
				continue
			gain_ratio= self.gain_ratio(attr_info_gain[attr], curr_avc_group[attr], len(data))
			if(gain_ratio > max_measure):
				best_attr = attr
				max_measure = gain_ratio
				
		if best_attr == 0:
			best_attr = self.choose_random_attribute(attribute_list)

		return best_attr
	
	
	def majority_vote_class(self, data):
		classs_labels = [tuple["class_label"] for tuple in data.values()]
		return max(classs_labels, key = classs_labels.count)
	
		
	def get_proj_data(self, data, attr_index, attr_value):		
		proj_data = {tuple_id : tuple_attr_list for tuple_id, tuple_attr_list in data.items() \
					if tuple_attr_list[attr_index] == attr_value}				
		return proj_data
	'''									
		*********************************************************************************
	'''



	'''	
		additional methods for Random Forest								
	'''
	def construct(self, attr_sample_num):
		self.pre_process_data()
		self._tree = self.generate_decision_tree(self._class_attribute_tuples, self._attribute_values_dict, attr_sample_num,"random forest", -1)		
	
	
	def sample_random_attribute_list(self, attribute_list, sample_num):		
		new_sample_num = int(math.sqrt(len(attribute_list))) if len(attribute_list) < sample_num else sample_num
		sample_indice = random.sample(attribute_list, new_sample_num)
		sample_list = {id:v for(id,v) in attribute_list.items() if id in sample_indice}
		return sample_list
		
	
	
	'''	
		main tree construction and classify methods								
	'''
	def generate_decision_tree(self, data, attribute_list, sample_num, mode, branch_depth):
		
		class_labels = [tuple["class_label"] for tuple in data.values()]
		
		#if all samples belong to same class, directly return the class label
		if class_labels.count(class_labels[0]) == len(class_labels):
			branch_depth += 1
			#print "depth:", branch_depth
			if branch_depth > self._curr_max_depth :
				self._curr_max_depth = branch_depth 
			return str(class_labels[0])
			
		#if attribute list is empty or reach maximal depth
		#return the majority class in current dataset
		if (not attribute_list) or (branch_depth + 1 >= self._max_depth):
			branch_depth += 1
			#print "depth:", branch_depth
			if branch_depth > self._curr_max_depth :
				self._curr_max_depth = branch_depth 
			return str(self.majority_vote_class(data))
			
		#otherwise, choose the "best" splitting attribute
		#and generate subtree recursively
		else:
			if mode == "single tree":
				best_attr = self.choose_best_attribute_gain_raio(data, attribute_list)
			elif mode == "random forest":		
				sample_attribute_list = self.sample_random_attribute_list(attribute_list, sample_num)		
				best_attr = self.choose_best_attribute_cart(data, sample_attribute_list)
			
			#print "best_attr is " + str(best_attr)
			
			sub_attribute_list = {attr: attribute_list[attr] for attr in attribute_list if attr != best_attr}
			
			tree = {best_attr : {}}
			
			for attr_value in attribute_list[best_attr]:
				proj_data = self.get_proj_data(data, best_attr, attr_value)
				new_branch_depth = branch_depth + 2
				if len(proj_data) < self._min_split_size:
					tree[best_attr][attr_value] = str(self.majority_vote_class(data))
					new_branch_depth += 1
				else:
					new_branch_depth += 2
					subtree = self.generate_decision_tree(proj_data, sub_attribute_list, sample_num, mode, new_branch_depth)
					tree[best_attr][attr_value] = subtree
				
				if new_branch_depth > self._curr_max_depth :
					self._curr_max_depth = new_branch_depth 
		
		return tree	
	
	
	def predict_class_label(self, tuple):
		return self.predict_class_label_recursive(self._tree,tuple)
		
			
	def predict_class_label_recursive(self, tree, tuple):
		if type(tree) == str: 
			return int(tree)
		else:
			#print tree
			#print tuple
			attr_index = tree.keys()[0]
			attr_value = tuple[attr_index]
			#print attr_index, attr_value
			if not attr_value in tree[attr_index]:
				split_value = random.sample(self._attribute_values_dict[attr_index],1)[0]
			else:
				split_value = attr_value
			return self.predict_class_label_recursive(tree[attr_index][split_value], tuple)
	
	
	def run_dataset(self, dataset):
		output_result = {"TP":0, "FN":0, "FP":0, "TN":0}
		
		for tuple_id, tuple_info in dataset.items():
			real_label = tuple_info["class_label"]
			predict_label = self.predict_class_label(tuple_info)
			
			if predict_label == 1:
				if real_label == 1:
					output_result["TP"] += 1
					print tuple_id
				else:
					output_result["FP"] += 1
			else:
				if real_label == -1:
					output_result["TN"] += 1
				else:
					output_result["FN"] += 1
		
		return output_result		
	
	
	def classify(self):
		self.pre_process_data()
		self._tree = self.generate_decision_tree(self._class_attribute_tuples, self._attribute_values_dict, -1, "single tree", -1)
		
		self._output_result = self.run_dataset(self._class_attribute_tuples)
		self._output_result_test = self.run_dataset(self._class_attribute_tuples_test)
		
		print self._output_result["TP"],self._output_result["FN"],self._output_result["FP"],self._output_result["TN"]
		print self._output_result_test["TP"],self._output_result_test["FN"],self._output_result_test["FP"],self._output_result_test["TN"]
	
		#self.debug_print_tree()
		self.debug_print_tree_recursively(self._tree, "")
		#self.debug_evaluate(self._output_result)
		#self.debug_evaluate(self._output_result_test)
		#print self._max_depth, self._curr_max_depth
	'''									
		*********************************************************************************
	'''
	
	
						
	'''
		debug methods
	'''				
	def debug_display_data(self):
		print"print training data:"
		for tuple_id, tuple_info in self._class_attribute_tuples.items():
				print tuple_id, tuple_info
			
		print "print test data:"
		for tuple_id_test, tuple_info_test in self._class_attribute_tuples_test.items():
				print tuple_id_test, tuple_info_test
				
					
	def debug_f_score(self, precision, recall, beta):
		return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
	
	
	def debug_evaluate(self, output_result):
		total = sum(output_result.values())
		evaluation_metrics = {}
		evaluation_metrics["Accuracy"] = float(output_result["TP"] + output_result["TN"]) / total
		evaluation_metrics["Error Rate"] = float(output_result["FP"] + output_result["FN"]) / total
		evaluation_metrics["Sensitivity"] = float(output_result["TP"]) / (output_result["TP"] + output_result["FN"])
		evaluation_metrics["Specificity"] = float(output_result["TN"]) / (output_result["TN"] + output_result["FP"])
		evaluation_metrics["Precision"] = float(output_result["TP"]) / (output_result["TP"] + output_result["FP"])
		evaluation_metrics["F-1 Score"] = self.debug_f_score(evaluation_metrics["Precision"], evaluation_metrics["Sensitivity"], 1)
		evaluation_metrics["F-0.5 Score"] = self.debug_f_score(evaluation_metrics["Precision"], evaluation_metrics["Sensitivity"], 0.5)
		evaluation_metrics["F-2 Score"] = self.debug_f_score(evaluation_metrics["Precision"], evaluation_metrics["Sensitivity"], 2)
		
		for (x,y) in evaluation_metrics.items():
			print x, y
	
		
	def debug_print_tree(self):
		print self._tree
			
	
	def debug_print_tree_recursively(self, tree, path):
		if type(tree) == str:
			print path +"c" + tree
		else:
			attr_index = tree.keys()[0]
			attr_values = self._attribute_values_dict[attr_index]
			for attr_value in attr_values:
			 	newpath = path + "attr"+str(attr_index) + "-" + "value"+str(attr_value)+"-"
				self.debug_print_tree_recursively(tree[attr_index][attr_value], newpath)
	'''									
		*********************************************************************************
	'''