using System.Drawing;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using Numpy;

namespace MachineLearning;

public class Tool
{

    private static readonly string TrainImagePath = @"C:\Code\Git\tf_not\Asset\mnist_png.tar\mnist_png\training";
    private static readonly string TestImagePath = @"C:\Code\Git\tf_not\Asset\mnist_png.tar\mnist_png\testing";
    private static readonly string train_date_path = @"C:\Test\cnn_train_data.bin";
    private static readonly string train_label_path = @"C:\Test\cnn_train_label.bin";
    private static readonly string ModelFile = @"C:\Test\cnn_fashion_mnist.h5";

    private static readonly int img_rows = 28;
    private static readonly int img_cols = 28;
    private static readonly int channel = 1;
    private static readonly int num_classes = 10;  // total classes

    /// <summary>
    /// 加载训练数据
    /// </summary>
    /// <param name="total_size"></param>    
    public static (NDarray, NDarray) LoadTrainingData()
    {
        return LoadRawData();
    }

    private static (NDarray, NDarray) LoadRawData()
    {
        Console.WriteLine("LoadRawData");

        int total_size = 60000;
        float[,,,] arrx = new float[total_size, img_rows, img_cols, channel];
        int[,] arry = new int[total_size, 10];

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
                        arry[count, int.Parse(Dir.Name)] = 1;
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

}