using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;

int batch_size = 128;
int num_classes = 10;
int epochs = 12;

// input image dimensions
int img_rows = 28, img_cols = 28;

Shape input_shape = null;

var ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();


if (keras.backend.image_data_format() == ImageDataFormat.channels_first)
{
    x_train = x_train.reshape((x_train.shape[0], 1, img_rows, img_cols));
    x_test = x_test.reshape((x_test.shape[0], 1, img_rows, img_cols));
    input_shape = (1, img_rows, img_cols);
}
else
{
    x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1));
    x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, 1));
    input_shape = (img_rows, img_cols, 1);
}


Console.ReadKey();