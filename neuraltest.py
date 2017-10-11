#!/usr/bin/ python3.5

########################################
# TODO                                 #
#                                      #
# Refactor each function               #
# Save weights into a file             #
# Implement the network as a library   #
########################################

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

#Class for the neural network
class neuralNetwork():
	#Initialization function for the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.onodes = outputnodes
		self.hnodes = hiddennodes
		self.lr = learningrate

		#Link weight matrices wih and who
		#Weights inside the array are w_i_j which go from node i to node j
		self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes, self.hnodes))
		self.activation_function = lambda x: scipy.special.expit(x)

	#Updates the value of the weights according to the training data
	def updateWeights(self, inputs_list, targets_list):
		#Converts inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		#Calculate signals into hidden layers
		hidden_inputs = np.dot(self.wih, inputs)
		#Calculate signals from hidden layers
		hidden_outputs = self.activation_function(hidden_inputs)
		#Calcualte signals into final output layer
		final_inputs = np.dot (self.who, hidden_outputs)
		#Calculate signals from final output layer
		final_outputs = self.activation_function(final_inputs)
		#Error is targets - actual_outputs
		output_errors = targets - final_outputs
		#Hidden layer error is the output_errors, split by weights
		hidden_errors = np.dot(self.who.T, output_errors)
		#Update the weights between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)) , np.transpose(hidden_outputs)) 
		#Update the weights between the hidden and input layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) , np.transpose(inputs))

	#Query function to obtain the network guess once it has been trained
	def query(self, inputs_list):
		#Converts imputs to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		#Calculate signals into hidden layers
		hidden_inputs = np.dot(self.wih, inputs)
		#Calculate signals emerging from hidden layers
		hidden_outputs = self.activation_function(hidden_inputs)
		#Calculate signals into final layer
		final_inputs = np.dot(self.who, hidden_outputs)
		#Calcualte signas emerging from output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	#Trains the network
	def train(self, epochs, data_training):
		#Handling the file for the training data
		training_data_file = open(data_training,'r')
		training_data_list = training_data_file.readlines()
		training_data_file.close()
		#Go through all the records
		for e in range(epochs):
			print("Running epoch #" + str(e+1) +".")
			for record in training_data_list:
				all_values = record.split(',')
				inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
				#Create the target values	
				targets = np.zeros(output_nodes) + 0.01
				#Value  0.99 is the target value
				targets[int(all_values[0])] = 0.99
				self.updateWeights(inputs, targets)
			print("Finished epoch #" + str(e) + ".")	


	#Test the neural network
	def test(self, data_test):
		#Handling the file for the test data
		test_data_file = open(data_test,'r')
		test_data_list = test_data_file.readlines()
		test_data_file.close()
		scores=[]
		print("Starting test process.")
		for record in test_data_list:
			all_values = record.split(',')
			correct_label = int(all_values[0])
			print(correct_label, "Correct label")
			#Scale and shift the inputs
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			#Query the network for the values
			outputs = self.query(inputs)
			guess = np.argmax(outputs)
			print(guess, "Network guess")
			if (guess == correct_label):
				scores.append(1)
			else:
				scores.append(0)
		#Calculate the performance
		scores_array = np.asarray(scores)
		print("Performance = ", scores_array.sum() / scores_array.size)

#Main code of the program
if __name__ == "__main__":
	input_nodes = 784
	hidden_nodes= 200
	output_nodes= 10
	learning_rate = 0.2
	epochs = 1
	train_file = "mnist_train.csv"
	test_file = "mnist_test.csv"

	net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
	net.train(epochs, train_file)
	net.test(test_file)
       

