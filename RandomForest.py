import sys
from RandomForestClass import *
from Auxilliary import *

training_file = sys.argv[1]
test_file = sys.argv[2]
			
aux = Auxilliary()
training_data, attribute_list_train = aux.load_data(training_file)
test_data, attribute_list_test = aux.load_data(test_file)

num_of_trees = 5
num_of_attribute = 11

random_forest = RandomForest(training_data, test_data, attribute_list_train, num_of_trees, num_of_attribute)
random_forest.classify()