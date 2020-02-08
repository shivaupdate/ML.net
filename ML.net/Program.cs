using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace Lab1
{
    class FeedBackTrainingData
    {
        [Column(ordinal: "0", name: "Label")]
        public bool IsGood { get; set; }

        [Column(ordinal: "1")]
        public string FeedBackText { get; set; }

       
    }

    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }
    class Program
    {
        static List<FeedBackTrainingData> trainingdata = 
            new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData =
           new List<FeedBackTrainingData>();
        static void LoadTestData()
        {
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "horrible terrible",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shit",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "sweet",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "average",
                IsGood = true
            });
        }
            static void LoadTrainingData()
        {
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText ="this is good",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "it very Average",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad horrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "well ok ok",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shitty terrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "soooo nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "cool nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "sweet and nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and good",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "very good",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "quiet average",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "soooo nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "god horrible",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "average and ok",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad and hell",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this nice but better can be done",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad bad",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "till now it looks nice",
                IsGood = true
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shit",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "oh this is shit",
                IsGood = false
            });

        }
        static void Main(string[] args)
        {
            //Step 1 :-  We need to load the training data
            LoadTrainingData();
           
            // Step 2 :- Create object of MLCOntext
            var mlContext = new MLContext();
            // Step 3 :- Convert your data in to IDataView
            IDataView dataView = mlContext.CreateStreamingDataView
                                    <FeedBackTrainingData>(trainingdata);
            // Step 4 :- We need to create the pipe line 
            // define the work flows in it.
            var pipeline = mlContext.Transforms.
                        Text.FeaturizeText("FeedBackText", "Features")
                        .Append(mlContext.BinaryClassification.Trainers.FastTree
                        (numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));
            // Step 5 :- Traing the algorithm and we want the model out
            var model = pipeline.Fit(dataView);
            // Step 6 :- Load the test data and run the test data
            // to check our models accuracy
            LoadTestData();
            IDataView dataView1 = mlContext.
                              CreateStreamingDataView<FeedBackTrainingData>(testData);

            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);

            // Step 7 :- use the model
            string strcont = "Y";
            while (strcont == "Y")
            {
                Console.WriteLine("Enter a feedback string");
                string feedbackstring = Console.ReadLine().ToString();
                var predictionFunction = model.MakePredictionFunction
                                           <FeedBackTrainingData, FeedBackPrediction>
                                           (mlContext);
                var feedbackinput = new FeedBackTrainingData();
                feedbackinput.FeedBackText = feedbackstring;
                var feedbackpredicted = predictionFunction.Predict(feedbackinput);
                Console.WriteLine("Predicted :- " + feedbackpredicted.IsGood);
            }
            Console.ReadLine();
        }
    }
}
