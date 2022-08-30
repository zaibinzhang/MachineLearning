using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

/*
参考网站
https://keras.io/examples/vision/mnist_convnet/

https://github.com/SciSharp/Keras.NET

*/
var numClasses = 10;
var inputShape = new Shape(28, 28, 1);

var ((xTrain, yTrain), (xTest, yTest)) = keras.datasets.mnist.load_data();
xTrain /= 255;
xTest /= 255;

xTrain = np.expand_dims(xTrain, -1);
xTest = np.expand_dims(xTest, -1);
Console.WriteLine("x_train shape:" + xTrain.shape);
Console.WriteLine(xTrain.shape[0] + " train samples");
Console.WriteLine(xTest.shape[0] + " test samples");

yTrain = np_utils.to_categorical(yTrain, numClasses);
yTest = np_utils.to_categorical(yTest, numClasses);

var model = keras.Sequential();
model.add(keras.Input(inputShape));
model.add(keras.layers.Conv2D(32, kernel_size: (3, 3), activation: keras.activations.Relu));
model.add(keras.layers.MaxPooling2D((2, 2)));
model.add(keras.layers.Conv2D(64, kernel_size: (3, 3), activation: keras.activations.Relu));
model.add(keras.layers.MaxPooling2D((2, 2)));
model.add(keras.layers.Flatten());
model.add(keras.layers.Dropout(0.5f));
model.add(keras.layers.Dense(numClasses, activation: keras.activations.Softmax));
model.compile(loss: keras.losses.CategoricalCrossentropy(), optimizer: keras.optimizers.Adam(), metrics: new[] { "accuracy" });

model.summary();

Console.ReadKey();