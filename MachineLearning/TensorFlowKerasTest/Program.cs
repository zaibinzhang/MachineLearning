using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

var model = keras.Sequential();

#region 官方例子




//Model model;
//NDArray x_train, y_train, x_test, y_test;

////tf.enable_eager_execution();
////PrepareData
//(x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
//x_train = x_train.reshape((60000, 784)) / 255f;
//x_test = x_test.reshape((10000, 784)) / 255f;

////BuildModel
//// input layer
//var inputs = keras.Input(shape: 784);

//// 1st dense layer
//var outputs = keras.layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);

//// 2nd dense layer
//outputs = keras.layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);

//// output layer
//outputs = keras.layers.Dense(10).Apply(outputs);

//// build keras model
//model = keras.Model(inputs, outputs, name: "mnist_model");
//// show model summary
//model.summary();

//// compile keras model into tensorflow's static graph
//model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
//    optimizer: keras.optimizers.RMSprop(),
//    metrics: new[] { "accuracy" });

//// train model by feeding data and labels.
//model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);

//// evluate the model
//model.evaluate(x_test, y_test, verbose: 2);

//// save and serialize model
//model.save("d:\\mnist_model.mod");

// recreate the exact same model purely from the file:
// model = keras.models.load_model("path_to_my_model");
#endregion

Console.ReadKey();