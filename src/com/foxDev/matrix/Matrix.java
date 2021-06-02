package com.foxDev.matrix;

import java.util.concurrent.ThreadLocalRandom;
import com.foxDev.neuralNetwork.NeuralNetwork;

public class Matrix {
	public int rows;
	public int cols;
	public double[][] data;
	
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		
		this.data = new double[this.rows][this.cols];
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.data[i][j] = 0;
			}
		}
	}
	
	public Matrix(double[][] data, int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		
		this.data = data;
	}
	
	public static Matrix fromArray(double[] arr) {
		Matrix m = new Matrix(arr.length,1);
		
		for(int i = 0; i < arr.length; i++) {
			m.data[i][0] = arr[i];
		}
		return m;
	}
	
	public void expandMatrix(int newCols) {
		double[][] newData = new double[this.rows][newCols];
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < newCols; j++) {
				newData[i][j] = this.data[i][0];
			}
		}
		this.data = newData;
		this.cols = newCols;
	}
	
	public static Matrix transpose(Matrix m) {
		//From size (Rows,Cols) to (Cols,Rows) while keeping data
		Matrix result = new Matrix(m.cols,m.rows);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.cols; j++) {
				result.data[j][i] = m.data[i][j];
			}
		}
		return result;
	}
	
	public void multiply(double n) {
		//Scalar product
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.data[i][j] *= n;
			}
		}
	}
	
	public static Matrix multiply(Matrix m, double n) {
		//Scalar product
		
		Matrix result = new Matrix(m.rows,m.cols);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.cols; j++) {
				result.data[i][j] = m.data[i][j] * n;
			}
		}
		return result;
	}
	
	public static Matrix multiply(Matrix a, Matrix b) {
		//Matrix Product
		if(a.cols != b.rows) { return null; } 
		
		double[][] result = new double[a.rows][b.cols];
		
		for(int i = 0; i < a.rows; i++) {
			for(int j = 0; j < b.cols; j++) {
				for(int k = 0; k < a.cols; k++) {
					result[i][j] += a.data[i][k] * b.data[k][j];
				}
			}
		}
		return new Matrix(result,a.rows,b.cols);
	}
	
	public Matrix multiply(Matrix m) {
		//Matrix Product
		if(this.cols != m.rows) { return null; } 
		
		Matrix result = new Matrix(this.rows,m.cols);
		
		for(int i = 0; i < result.rows; i++) {
			for(int j = 0; j < result.cols; j++) {
				float sum = 0;
				for(int k = 0; k < this.cols; k++) {
					sum += this.data[i][k] * m.data[k][j];
				}
				result.data[i][j] = sum;
			}
		}
		return result;
	}
	
	public void add(double n) {
		//Scalar addition
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.data[i][j] += n;
			}
		}
	}
	
	public static Matrix add(Matrix m, double n) {
		//Scalar addition
		
		Matrix result = new Matrix(m.rows,m.cols);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.cols; j++) {
				result.data[i][j] = m.data[i][j] + n;
			}
		}
		return result;
	}
	
	public void add(Matrix n) {
		//Element-wise addition
		if(this.rows != n.rows || this.cols != n.cols) { return; }
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.data[i][j] += n.data[i][j];
			}
		}
	}
	
	public static Matrix add(Matrix a, Matrix b) {
		//Element-wise addition
		if(a.rows != b.rows || a.cols != b.cols) { return null; }
		
		Matrix result = new Matrix(a.rows,a.cols);
		
		for(int i = 0; i < a.rows; i++) {
			for(int j = 0; j < a.cols; j++) {
				result.data[i][j] = a.data[i][j] + b.data[i][j];
			}
		}
		return result;
	}
	
	public void randomize() {
		//Randomize values
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.data[i][j] = ThreadLocalRandom.current().nextFloat()*2-1;
			}
		}
	}
	
	public void print() {
		//Print to console
        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < this.data[i].length; j++) {
                System.out.print(String.valueOf(this.data[i][j]) + " ");
            }
            System.out.print("\n");
        }
        System.out.print("\n");
    }
	
	public static Matrix subtract(Matrix a, Matrix b) {
		if(a.rows != b.rows || a.cols != b.cols) { return null; }
		
		Matrix result = new Matrix(a.rows,a.cols);
		for(int i = 0; i < a.rows; i++) {
			for(int j = 0; j < a.cols; j++) {
				result.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
		return result;
	}
	
	public static void main(String args[]) {
		/*Matrix a = new Matrix(2,3);
		Matrix b = new Matrix(3,2);
		a.randomize();
		b.randomize();
		a.print();
		b.print();
		Matrix c = multiply(a,b);
		c.print();*/
		
		//Matrix a = new Matrix(2,3);
		//a.randomize();
		//a.print();
		//System.out.print(a.toArray().toString());
		
		NeuralNetwork nn = new NeuralNetwork(2,2,1);
		double[][] training_inputs = {
				{0,1},
				{1,0},
				{1,1},
				{0,0}
		};
		double[][] training_targets = {
				{1},
				{1},
				{0},
				{0}
		};
		for(int i = 0; i < 100; i++) {
			int index = ThreadLocalRandom.current().nextInt(0,3);
			nn.train(training_inputs[index],training_targets[index]);
		}
		nn.guess(training_inputs[0]).print();
		nn.guess(training_inputs[1]).print();
		nn.guess(training_inputs[2]).print();
		nn.guess(training_inputs[3]).print();
	}
}
