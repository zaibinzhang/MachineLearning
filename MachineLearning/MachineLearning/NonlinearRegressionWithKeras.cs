using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace MachineLearning
{
    /// <summary>
    /// 用神经网络实现非线性回归
    /// </summary>
    public class NonlinearRegressionWithKeras
    {
        private readonly Random random = new Random(1);

        public void Run()
        {
            //1、创建模型
            Model model = BuildModel();
            model.compile(loss: keras.losses.MeanSquaredError(),
                optimizer: keras.optimizers.SGD(0.02f),
                metrics: new[] { "mae" }); // acc( accuracy):准确性 或 mae:平均绝对误差(Mean absolute Error)
            model.summary();



            //2、训练模型
            (NDArray train_x, NDArray train_y) = PrepareData(1000);
            model.fit(train_x, train_y, batch_size: 64, epochs: 100);

            //3、应用模型（消费）
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
            int num_features = 1; // data features 
            int n_hidden_1 = 16; // 1st layer number of neurons.     
            int num_out = 1; // total output .

            var model = keras.Sequential();
            model.add(keras.Input(num_features));
            model.add(keras.layers.Dense(n_hidden_1));
            model.add(keras.layers.LeakyReLU(0.2f));
            model.add(keras.layers.Dense(num_out));

            return model;
        }

        /// <summary>
        /// 加载训练数据
        /// </summary>
        /// <param name="total_size"></param>    
        private (NDArray, NDArray) PrepareData(int total_size)
        {
            float[,] arrx = new float[total_size, 1];
            float[] arry = new float[total_size];

            for (int i = 0; i < total_size; i++)
            {
                float x = (float)random.Next(-400, 400) / 100;
                float y = x * x;

                arrx[i, 0] = x;
                arry[i] = y;
            }

            NDArray train_X = np.array(arrx);
            NDArray train_Y = np.array(arry);

            return (train_X, train_Y);
        }

        /// <summary>
        /// 消费模型
        /// </summary>      
        private void test(Model model)
        {
            int test_size = 10;

            for (int i = 0; i < test_size; i++)
            {
                float x = (float)random.Next(-300, 300) / 100;
                float y = x * x;

                var test_x = np.array(new float[1, 1] { { x } });
                var pred_y = model.Apply(test_x);

                Console.WriteLine($"{i}:x={(float)test_x:0.00}\ty={y:0.0000} Pred:{(float)pred_y[0].numpy():0.0000}");
            }
        }
    }
}
