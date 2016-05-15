package com.chaturv.alerts;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Vineet on 15/5/2016.
 */
public class DBNClassifyAlerts {

    private static Logger log = LoggerFactory.getLogger(DBNClassifyAlerts.class);

    private static int numSamples = 150;
    private static int splitTrainNum = (int) (numSamples * .8);
    private static int seed = 123;
    private static int iterations = 1000; //number of times you allow a net to classify samples and be corrected with a weight update
    private static int listenerFreq = iterations/50;

    private static int numRows = 4;       //input layer rows
    private static int numColumns = 1;    //what is this?
    private static int outputNum = 3;

    public static void main(String[] args) throws IOException, InterruptedException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        DBNClassifyAlerts classifier = new DBNClassifyAlerts();

//        SplitDataSet splitDataSet = classifier.loadData();
        SplitDataSet splitDataSet = classifier.loadDataIris();
        classifier.train(splitDataSet);
    }

    private void train(SplitDataSet splitDataSet) {
        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed) // Locks in weight initialization for tuning
            .iterations(iterations) // # training iterations predict/classify & backprop
            .learningRate(1e-6) //size of the adjustments made to the weights with each iteration
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients

//            .l1(1e-1) //L1 regularization coefficient
            .l2(2e-4) //L2 regularization coefficient
            .regularization(true)
            .momentum(0.9)
            .useDropConnect(true)
            .list(2) // # NN layers (doesn't count input layer)

            .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                .nIn(numRows * numColumns)
                .nOut(3) //# fully connected hidden layer nodes. Add list if multiple layers.
                .weightInit(WeightInit.XAVIER) // Weight initialization
                .k(1)
                .activation("relu") // Activation function type
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                .updater(Updater.ADAGRAD)
                .dropOut(0.5)
                .build()
            )
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .nIn(3) //# input nodes
                .nOut(outputNum) // # output nodes
                .activation("softmax")
                .build()
            )
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(listenerFreq));

        System.out.println("Train model....");
        model.fit(splitDataSet.train);

        System.out.println("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            System.out.println("Weights: " + w);
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        eval.eval(splitDataSet.test.getLabels(), model.output(splitDataSet.test.getFeatureMatrix(), Layer.TrainingMode.TEST));
        System.out.println(eval.stats());


        log.info("**************** Done. Writing to file ********************");
    }


    private SplitDataSet loadData() throws IOException, InterruptedException {
        //TODO: add bias unit? See IrisUtils. INDArray ret = Nd4j.ones(Math.abs(to - from), 4);
        InputSplit inputSplit = new FileSplit(new File("C:/work/repo/dl4j-0.4-examples/src/main/resources/alerts_distinct_clusters.csv"));

        //create Canova record reader
        RecordReader recordReader = new CSVRecordReader(1); //skip header
        recordReader.initialize(inputSplit);

        //read the entire dataset
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, numSamples, 2, 3); //output label idx, numPossibleLabels
//        while (iter.hasNext()) {
        DataSet dataSet = iter.next();
        dataSet.shuffle();
//        dataSet.normalizeZeroMeanZeroUnitVariance();
//        }

        //print
        System.out.println(dataSet);

        //split
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        //print
        System.out.println(train);
        System.out.println(test);

        return new SplitDataSet(train, test);
    }

    private SplitDataSet loadDataIris() throws IOException, InterruptedException {
        //read the entire dataset
        DataSetIterator iter = new IrisDataSetIterator(numSamples, numSamples);

        DataSet dataSet = iter.next();
        dataSet.shuffle();
        dataSet.normalizeZeroMeanZeroUnitVariance();


        //print
        System.out.println(dataSet);

        //split
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        //print
        System.out.println(train);
        System.out.println(test);

        return new SplitDataSet(train, test);
    }

    /**
     * container class
     */
    private class SplitDataSet {
        DataSet train;
        DataSet test;

        SplitDataSet(DataSet train, DataSet test) {
            this.train = train;
            this.test = test;
        }
    }

}
