using Tensorflow.Keras.Optimizers;

namespace TensorFlowKerasTest;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using Tensorflow.Operations.Losses;
using static System.Formats.Asn1.AsnWriter;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

public class CopyKerasNetTestFunction
{
    public static void Run()
    {
        int batchSize = 200;
        int numClasses = 10;
        int epochs = 10;

        // input image dimensions
        int imgRows = 28, imgCols = 28;
        Shape input_shape = null;

        var ((xTrain, yTrain), (xTest, yTest)) = keras.datasets.mnist.load_data();

        var format = keras.backend.image_data_format();

        if (format == ImageDataFormat.channels_first)
        {
            xTrain = xTrain.reshape(new(xTrain.shape[0], 1, imgRows, imgCols));
            xTest = xTest.reshape(new(xTest.shape[0], 1, imgRows, imgCols));
            input_shape = (1, imgRows, imgCols);
        }
        else
        {
            xTrain = xTrain.reshape(new(xTrain.shape[0], imgRows, imgCols, 1));
            xTest = xTest.reshape(new(xTest.shape[0], imgRows, imgCols, 1));
            input_shape = (imgRows, imgCols, 1);
        }

        xTrain = xTrain.astype(np.float32);
        xTest = xTest.astype(np.float32);
        xTrain /= 255;
        xTest /= 255;
        Console.WriteLine("x_train shape: " + xTrain.shape);
        Console.WriteLine(xTrain.shape[0] + " train samples");
        Console.WriteLine(xTest.shape[0] + " test samples");

        // convert class vectors to binary class matrices
        yTrain = np_utils.to_categorical(yTrain, numClasses);
        yTest = np_utils.to_categorical(yTest, numClasses);

        // Build CNN model
        var inputs = keras.Input(input_shape);

        var outputs = keras.layers.Conv2D(32, kernel_size: (3, 3), activation: keras.activations.Relu).Apply(inputs);

        outputs = keras.layers.Conv2D(64, kernel_size: (3, 3), activation: keras.activations.Relu).Apply(outputs);

        outputs = keras.layers.MaxPooling2D((2, 2)).Apply(outputs);

        outputs = keras.layers.Dropout(0.25f).Apply(outputs);

        outputs = keras.layers.Flatten().Apply(outputs);

        outputs = keras.layers.Dense(128, activation: keras.activations.Relu).Apply(outputs);

        outputs = keras.layers.Dropout(0.5f).Apply(outputs);

        outputs = keras.layers.Dense(numClasses, activation: keras.activations.Softmax).Apply(outputs);

        var model = keras.Model(inputs, outputs, name: "mnist_model");

        model.summary();

        model.compile(loss: keras.losses.CategoricalCrossentropy(), optimizer: new Adam(), metrics: new string[] { "accuracy" });

        model.fit(xTrain, yTrain,
            batch_size: batchSize,
            epochs: epochs,
            verbose: 1);


        Console.ReadKey();
    }
}