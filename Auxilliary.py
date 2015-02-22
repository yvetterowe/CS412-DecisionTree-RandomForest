import sys
import re

class Auxilliary(object):
	def __init__(self):
		pass
	
		
	def load_data(self, data_file):
		class_attribute_tuples = {}
		attribute_values_dict = {}
		attribute_list = set()

		FILE = open(data_file, "r")
		
		tuple_id = 0
		for line in FILE:
			if not line.strip():
				continue
				
			tuple = line.split()
					
			#load class label
			class_label = int(tuple[0])
			class_attribute_tuples[tuple_id] = {}
			class_attribute_tuples[tuple_id]["class_label"] = class_label
			
			#load attribute index and attribute value
			for i in range(1, len(tuple)):
				attr_index_value = re.split('\:+', tuple[i])
				attr_index = int(attr_index_value[0])
				attr_value = int(attr_index_value[1])	
						
				class_attribute_tuples[tuple_id][attr_index] = attr_value
				attribute_list.update([attr_index])
				
				'''if not attr_index in attribute_values_dict:
					attribute_values_dict[attr_index] = set()
				attribute_values_dict[attr_index].update([attr_value])'''
							
			tuple_id += 1
		
		# set missing attribute value to 0 by default
		for tuple_id, tuple_info in class_attribute_tuples.items():
			for attr_index in attribute_list:
				if not attr_index in tuple_info:
					tuple_info[attr_index] = 0
					#attribute_values_dict[attr_index].update([0])
					
				
		FILE.close()
		#print attribute_list
		return class_attribute_tuples, attribute_list