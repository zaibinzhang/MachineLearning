using System.Drawing;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace MachineLearning;


/// <summary>
/// 采用卷积神经网络处理Fashion-MNIST数据集
/// </summary>
public class CNN_Fashion_MNIST
{
    private readonly string TrainImagePath = @"C:\Code\Git\tf_not\Asset\mnist_png.tar\mnist_png\training";
    private readonly string TestImagePath = @"C:\Code\Git\tf_not\Asset\mnist_png.tar\mnist_png\testing";
    private readonly string train_date_path = @"C:\Test\cnn_train_data.bin";
    private readonly string train_label_path = @"C:\Test\cnn_train_label.bin";
    private readonly string ModelFile = @"C:\Test\cnn_fashion_mnist.h5";

    private readonly int img_rows = 28;
    private readonly int img_cols = 28;
    private readonly int channel = 1;
    private readonly int num_classes = 10;  // total classes

    public void Run()
    {
        var model = BuildModel();
        model.summary();
        model.load_weights(ModelFile);


        model.compile(optimizer: keras.optimizers.Adam(0.0001f),
            loss: keras.losses.SparseCategoricalCrossentropy(),
            metrics: new[] { "accuracy" });

        (NDArray train_x, NDArray train_y) = LoadTrainingData();
        model.fit(train_x, train_y, batch_size: 512, epochs: 1);
        model.save_weights(ModelFile);

        test(model);
        Console.WriteLine("press any key");
        Console.ReadKey();
    }

    /// <summary>
    /// 构建网络模型
    /// </summary>     
    private Model BuildModel()
    {
        // 网络参数          
        int n_hidden_1 = 128;    // 1st layer number of neurons.     
        int n_hidden_2 = 128;    // 2nd layer number of neurons.                                
        float scale = 1.0f / 255;

        var model = keras.Sequential(new List<ILayer>
        {
            keras.layers.InputLayer((img_rows,img_cols)),
            keras.layers.Flatten(),
            keras.layers.Rescaling(scale),
            keras.layers.Dense(n_hidden_1, activation:keras.activations.Relu),
            keras.layers.Dense(n_hidden_2, activation:keras.activations.Relu),
            keras.layers.Dense(num_classes, activation:keras.activations.Softmax)
        });

        return model;
    }

    /// <summary>
    /// 加载训练数据
    /// </summary>
    /// <param name="total_size"></param>    
    private (NDArray, NDArray) LoadTrainingData()
    {
        try
        {
            Console.WriteLine("Load data");
            IFormatter serializer = new BinaryFormatter();
            FileStream loadFile = new FileStream(train_date_path, FileMode.Open, FileAccess.Read);
            float[,,,] arrx = serializer.Deserialize(loadFile) as float[,,,];

            loadFile = new FileStream(train_label_path, FileMode.Open, FileAccess.Read);
            int[] arry = serializer.Deserialize(loadFile) as int[];
            Console.WriteLine("Load data success");
            return (np.array(arrx), np.array(arry));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Load data Exception:{ex.Message}");
            return LoadRawData();
        }
    }

    private (NDArray, NDArray) LoadRawData()
    {
        Console.WriteLine("LoadRawData");

        int total_size = 60000;
        float[,,,] arrx = new float[total_size, img_rows, img_cols, channel];
        int[] arry = new int[total_size];

        int count = 0;

        DirectoryInfo RootDir = new DirectoryInfo(TrainImagePath);
        foreach (var Dir in RootDir.GetDirectories())
        {
            foreach (var file in Dir.GetFiles("*.png"))
            {

                Bitmap bmp = (Bitmap)Image.FromFile(file.FullName);
                if (bmp.Width != img_cols || bmp.Height != img_rows)
                {
                    continue;
                }

                for (int row = 0; row < img_rows; row++)
                    for (int col = 0; col < img_cols; col++)
                    {
                        var pixel = bmp.GetPixel(col, row);
                        int val = (pixel.R + pixel.G + pixel.B) / 3;
                        arrx[count, row, col, 0] = val;
                        arry[count] = int.Parse(Dir.Name);
                    }

                count++;
            }

            Console.WriteLine($"Load image data count={count}");
        }

        Console.WriteLine("LoadRawData finished");
        //Save Data
        Console.WriteLine("Save data");
        IFormatter serializer = new BinaryFormatter();

        //开始序列化
        FileStream saveFile = new FileStream(train_date_path, FileMode.Create, FileAccess.Write);
        serializer.Serialize(saveFile, arrx);
        saveFile.Close();

        saveFile = new FileStream(train_label_path, FileMode.Create, FileAccess.Write);
        serializer.Serialize(saveFile, arry);
        saveFile.Close();
        Console.WriteLine("Save data finished");

        return (np.array(arrx), np.array(arry));
    }

    /// <summary>
    /// 消费模型
    /// </summary>      
    private void test(Model model)
    {
        Random rand = new Random(1);

        DirectoryInfo TestDir = new DirectoryInfo(TestImagePath);
        foreach (var ChildDir in TestDir.GetDirectories())
        {
            Console.WriteLine($"Folder:【{ChildDir.Name}】");
            var Files = ChildDir.GetFiles("*.png");
            for (int i = 0; i < 10; i++)
            {
                int index = rand.Next(Files.Length - 1);
                var image = Files[index];

                var x = LoadImage(image.FullName);
                var pred_y = model.Apply(x);
                var result = argmax(pred_y[0].numpy());

                Console.WriteLine($"FileName:{image.Name}\tPred:{result}");
            }
        }
    }

    private NDArray LoadImage(string filename)
    {
        float[,,,] arrx = new float[1, img_rows, img_cols, channel];
        Bitmap bmp = (Bitmap)Image.FromFile(filename);

        for (int row = 0; row < img_rows; row++)
            for (int col = 0; col < img_cols; col++)
            {
                var pixel = bmp.GetPixel(col, row);
                int val = (pixel.R + pixel.G + pixel.B) / 3;
                arrx[0, row, col, 0] = val;
            }

        return np.array(arrx);
    }

    private int argmax(NDArray array)
    {
        var arr = array.reshape(-1);

        float max = 0;
        for (int i = 0; i < 10; i++)
        {
            if (arr[i] > max)
            {
                max = arr[i];
            }
        }

        for (int i = 0; i < 10; i++)
        {
            if (arr[i] == max)
            {
                return i;
            }
        }

        return 0;
    }
}