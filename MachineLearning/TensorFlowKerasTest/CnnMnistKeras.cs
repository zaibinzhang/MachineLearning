namespace TensorFlowKerasTest;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using static System.Formats.Asn1.AsnWriter;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
public class CnnMnistKeras
{
    public static void Run()
    {

        /*
        参考网站
        https://keras.io/examples/vision/mnist_convnet/

        https://github.com/SciSharp/SciSharp-Stack-Examples

        */


        var numClasses = 10;
        var inputShape = new Shape(28, 28, 1);

        #region 训练数据

        var (xTrain, yTrain, xTest, yTest) = keras.datasets.mnist.load_data();
        xTrain /= 255f;
        xTest /= 255f;

        xTrain = np.expand_dims(xTrain, -1);
        xTest = np.expand_dims(xTest, -1);
        Console.WriteLine("x_train shape:" + xTrain.shape);
        Console.WriteLine(xTrain.shape[0] + " train samples");
        Console.WriteLine(xTest.shape[0] + " test samples");

        yTrain = np_utils.to_categorical(yTrain, numClasses);
        yTest = np_utils.to_categorical(yTest, numClasses);

        #endregion

        #region 神经网络模型

        var inputs = keras.Input(inputShape);

        var outputs = keras.layers.Conv2D(32, kernel_size: (3, 3), activation: keras.activations.Relu).Apply(inputs);

        outputs = keras.layers.MaxPooling2D((2, 2)).Apply(outputs);

        outputs = keras.layers.Conv2D(64, kernel_size: (3, 3), activation: keras.activations.Relu).Apply(outputs);

        outputs = keras.layers.MaxPooling2D((2, 2)).Apply(outputs);

        outputs = keras.layers.Flatten().Apply(outputs);

        outputs = keras.layers.Dropout(0.5f).Apply(outputs);

        outputs = keras.layers.Dense(numClasses, activation: keras.activations.Softmax).Apply(outputs);

        var model = keras.Model(inputs, outputs, name: "mnist_model");

        model.summary();

        #endregion

        #region 训练模型

        var batchSize = 128;
        var epochs = 15;

        model.compile(loss: keras.losses.CategoricalCrossentropy(), optimizer: keras.optimizers.Adam(), metrics: new[] { "accuracy" });

        model.fit(xTrain, yTrain, batchSize, epochs, validation_split: 0.1f);

        #endregion

        #region 评估模型

        model.evaluate(xTest, yTest, verbose: 0);


        #endregion

        Console.ReadKey();
    }
}