using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Learners;
using System.Linq;

namespace mltest
{
    public class FAQData{
        [Column("0")][ColumnName("Question")]
        public string Question {get; set;}
        [Column("1")]
        [ColumnName("Label")]
        public int Answer {get;set;}
    }

    public class AnswerPrediction
    {
        [ColumnName("Label")]
        public int Answer;
        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
    class Program
    {
        static MLContext mlContext;
        static TransformerChain<MulticlassPredictionTransformer<MulticlassLogisticRegressionPredictor>> model;
        static void Main(string[] args)
        {
            TrainModel();
            Console.WriteLine("Model trained!");
            var question = "";
            while (question != "quit") {
                Console.WriteLine("What is your question?");
                question = Console.ReadLine();
                var answer = GetAnswer(question);

                Console.WriteLine(answer);
            }

        }

        static void TrainModel()
        {
            mlContext = new MLContext();

            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[]
                {
                    new TextLoader.Column("Question", DataKind.Text, 0),
                    new TextLoader.Column("Label", DataKind.Num, 1)
                },
                HasHeader = false,
                Separator = ","
            });

            var data = reader.Read("HRFAQ.csv");

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Question", "Features")
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(label: "Label"));

            
            model = pipeline.Fit(data);
        }
        static string GetAnswer(string Question)
        {
            var predictionFunc = model.MakePredictionFunction<FAQData, AnswerPrediction>(mlContext);
            
            var q = new FAQData()
            {
                Question = Question
            };

            AnswerPrediction prediction = predictionFunc.Predict(q);

            foreach (float score in prediction.Score)
            {
                Console.WriteLine(score * 100);
            }

            float maxValue = prediction.Score.Max();
            Console.WriteLine("guess percentage: " + (maxValue * 100));
            return prediction.Score.ToList().IndexOf(maxValue).ToString();

        }
}
}
