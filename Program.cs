using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;

using System.Linq;
using System.Collections.Generic;
using System.IO;


namespace mltest
{
    public class FAQData{
        [ColumnName("Question")]
        [LoadColumn(0)]
        public string Question {get; set;}
       
        [ColumnName("Label")]
        [LoadColumn(1)]
        public string Answer {get;set;}
    }

    public class AnswerPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Answer;
        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
    class Program
    {
        static MLContext mlContext;

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

            var data = mlContext.Data.LoadFromTextFile<FAQData>("Datasets/CrestFAQ2.csv", hasHeader: false, separatorChar: ',', allowSparse: false, allowQuoting: true, trimWhitespace:false);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Labels", inputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Question"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Labels", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "Labels"));

            var model = pipeline.Fit(data);
            
            using(FileStream stream = new FileStream("Models/crestmodel.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                Console.WriteLine("Saving File");
                mlContext.Model.Save(model,data.Schema,stream);
            }
        }
        static string GetAnswer(string Question)
        {
            DataViewSchema data = null;
            ITransformer model = null;
            using(FileStream stream = new FileStream("Models/crestmodel.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                Console.WriteLine("Loading file");
                model = mlContext.Model.Load(stream, out data);
            }

            var predictionFunc = mlContext.Model.CreatePredictionEngine<FAQData, AnswerPrediction>(model);
            
            var q = new FAQData()
            {
                Question = Question
            };

            AnswerPrediction prediction = predictionFunc.Predict(q);
            VBuffer<ReadOnlyMemory<char>> slotNames = new VBuffer<ReadOnlyMemory<char>>();
            predictionFunc.OutputSchema["Score"].GetSlotNames(ref slotNames);
            float maxValue = prediction.Score.Max();
            int index = prediction.Score.ToList().IndexOf(maxValue);

            foreach (float score in prediction.Score)
            {
                Console.WriteLine(score * 100);
            }


            Console.WriteLine("guess percentage: " + (maxValue * 100));
            Console.WriteLine(prediction.Answer);
            return slotNames.GetItemOrDefault(index).ToString();

        }
    }
}
