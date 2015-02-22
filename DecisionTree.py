import sys
from DecisionTreeClass import *
from Auxilliary import *

training_file = sys.argv[1]
test_file = sys.argv[2]
			
aux = Auxilliary()
training_data, attribute_list_train = aux.load_data(training_file)
test_data, attribute_list_test = aux.load_data(test_file)

max_depth = 200
min_split_size = 1

decision_tree = DecisionTree(training_data, test_data, attribute_list_train, max_depth, min_split_size)
decision_tree.classify()