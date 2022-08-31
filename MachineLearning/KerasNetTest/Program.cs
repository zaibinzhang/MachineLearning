// See https://aka.ms/new-console-template for more information
using Keras.Datasets;

var ((xTrain, yTrain), (xTest, yTest)) = MNIST.LoadData();

Console.ReadKey();