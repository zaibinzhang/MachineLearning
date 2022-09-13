using Keras.Datasets;
using Numpy;
using K = Keras.Backend;
using Keras;
using Keras.Models;
using Keras.Layers;
using Keras.Utils;
using Keras.Optimizers;

int batchSize = 200;
int numClasses = 10;
int epochs = 10;

// input image dimensions
int imgRows = 28, imgCols = 28;

Shape input_shape = null;

var ((xTrain, yTrain), (xTest, yTest)) = MNIST.LoadData();

var format = K.ImageDataFormat();

if (format == "channels_first")
{
    xTrain = xTrain.reshape(xTrain.shape[0], 1, imgRows, imgCols);
    xTest = xTest.reshape(xTest.shape[0], 1, imgRows, imgCols);
    input_shape = (1, imgRows, imgCols);
}
else
{
    xTrain = xTrain.reshape(xTrain.shape[0], imgRows, imgCols, 1);
    xTest = xTest.reshape(xTest.shape[0], imgRows, imgCols, 1);
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
yTrain = Util.ToCategorical(yTrain, numClasses);
yTest = Util.ToCategorical(yTest, numClasses);

// Build CNN model
var model = new Sequential();
model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
    activation: "relu",
    input_shape: input_shape));
model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
model.Add(new Dropout(0.25));
model.Add(new Flatten());
model.Add(new Dense(128, activation: "relu"));
model.Add(new Dropout(0.5));
model.Add(new Dense(numClasses, activation: "softmax"));

model.Compile(loss: "categorical_crossentropy",
    optimizer: new Adadelta(), metrics: new string[] { "accuracy" });

model.Fit(xTrain, yTrain,
    batch_size: batchSize,
    epochs: epochs,
    verbose: 1,
    validation_data: new NDarray[] { xTest, yTest });

//model.Save("model.h5");
//model.SaveTensorflowJSFormat("./");

var score = model.Evaluate(xTest, yTest, verbose: 0);
Console.WriteLine("Test loss:" + score[0]);
Console.WriteLine("Test accuracy:" + score[1]);

Console.ReadKey();