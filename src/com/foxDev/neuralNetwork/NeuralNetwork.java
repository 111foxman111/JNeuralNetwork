package com.foxDev.neuralNetwork;

import com.foxDev.matrix.Matrix;

public class NeuralNetwork {
	//Dimensions of network
	private int inputNodes;
	private int hiddenNodes;
	private int outputNodes;
	
	//Algorithm Data
	private Matrix[] weights = new Matrix[2];
	private Matrix[] biases = new Matrix[2];
	
	private double learningRate = 0.2;
	
	//Constructor
	public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
		//Set shape of network
		this.inputNodes = inputNodes;
		this.hiddenNodes = hiddenNodes;
		this.outputNodes = outputNodes;
		
		//Initialize network parameters
		initializeWeights();
		initializeBiases();
	}
	
	private void initializeWeights() {
		weights[0] = new Matrix(hiddenNodes, inputNodes);
	    weights[1] = new Matrix(outputNodes, hiddenNodes);
			
	    weights[0].randomize();
	    weights[1].randomize();
	}
	
	private void initializeBiases() {
		biases[0] = new Matrix(hiddenNodes,inputNodes);
		biases[1] = new Matrix(outputNodes,hiddenNodes);
		
		biases[0].randomize();
		biases[1].randomize();
	}
	
	//Guess method: one column matrix in -> one column matrix out
	public Matrix guess(double[] input) {
		if(input.length != inputNodes) {
			return null; //add error message here
		} else {
			//Transform array to matrix
			Matrix output = Matrix.fromArray(input);
			
			for (int i = 0; i < 2; i++) {
                output = calculateLayer(weights[i], biases[i], output);
            }
			return output;
		}
	}
	
	public void train(double[] inputArray, double[] targetArray) {
		if (inputArray.length != inputNodes) { return; }
		else if (targetArray.length != outputNodes) { return; }
		else {
			Matrix target = Matrix.fromArray(targetArray);
			Matrix input = Matrix.fromArray(inputArray);
			
			//Get all layers
			Matrix[] layers = new Matrix[3];
			layers[0] = input;
			for (int j = 1; j < layers.length; j++) {
                layers[j] = calculateLayer(weights[j - 1], biases[j - 1], layers[j - 1]);
                input = layers[j];
            }
			
			input.expandMatrix(2);
			
			for (int n = layers.length - 1; n > 0; n--) {
                // Calculate error
                Matrix errors = Matrix.subtract(layers[n],target); //layers[2] is weird

                // Calculate gradient
                Matrix gradients = calculateGradient(layers[n], errors);

                // Calculate delta
                Matrix deltas = calculateDeltas(gradients, layers[n - 1]); //Error 2 source: deltas returns null, layers[0] should be 2x2 but is 2x1

                // Apply gradient to bias
                biases[n - 1].add(gradients); //Error 1: mixed dims biases[0] should be 2x2 but is 2x1, grads is 2x2 and should be.

                // Apply delta to weights
                weights[n - 1].add(deltas); //Error 2: on n=1 deltas is null

                // Calculate and set target for previous (next) layer
                
                Matrix previousError = Matrix.transpose(weights[n-1]);
                previousError = Matrix.multiply(previousError,errors);
                target = Matrix.add(previousError,layers[n - 1]);
            }
		}
	} 
	
	private Matrix calculateLayer(Matrix weights, Matrix biases, Matrix input) {
		Matrix result = weights.multiply(input);
		result.add(biases);
		return sigmoidForMatrix(result);
	}
	
	private Matrix calculateGradient(Matrix layer, Matrix error) {
        Matrix gradient = sigmoidForMatrix(layer);
        gradient = Matrix.multiply(gradient,Matrix.transpose(error));
        gradient = Matrix.multiply(gradient,learningRate);
        return gradient;
    }
	
	private Matrix calculateDeltas(Matrix gradient, Matrix layer) {
        return Matrix.multiply(gradient,Matrix.transpose(layer));
    }
	
	private Matrix sigmoidForMatrix(Matrix input) {
		Matrix output = new Matrix(input.rows,input.cols);
		for(int i = 0; i < input.rows; i++) {
			for(int j = 0; j < input.cols; j++) {
				output.data[i][j] = sigmoid(input.data[i][j]);
			}
		}
		return output;
	}
		
	private double sigmoid(double x) {
		return 1/(1+Math.exp(-x));
	}
	
	//Testing starts here
	public static void main(String[] args) {
		double[] data = {0,1};
		
		NeuralNetwork nn = new NeuralNetwork(2,2,1);
		nn.guess(data).print();
    }
}
