package org.project.group2;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class MyProject {
    private static final Logger log = LoggerFactory.getLogger(MyProject.class);

    private static int seed = 123;
    private static int nClasses = 2;
    private static List<String> labels;

    // parameters for the Yolo2OutputLayer
    private static int nBoxes = 5;
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;
    private static double detectionThreshold = 0.2;
    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};

    // hyperparameters for training
    private static int batchSize = 2;
    private static int nEpochs = 50;
    private static double learningRate = 1e-4;

    // model
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/DrawingObjectDetection_tinyyolo.zip");
    private static ComputationGraph model;

    // bounding boxes
    private static Frame frame = null;
    private static final Scalar RED = RGB(255, 0, 0);
    private static final Scalar GREEN = RGB(0, 255, 0);
    private static Scalar[] colormap = {RED, GREEN};
    private static String labeltext = null;

    public static void main(String[] args) throws Exception {
        // preparing dataset
        org.project.group2.CarIterator.setup();
        RecordReaderDataSetIterator trainIter = org.project.group2.CarIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = org.project.group2.CarIterator.testIterator(1);
        labels = trainIter.getLabels();

        // check if there is saved trained model inside generated-models folder
        if (modelFilename.exists()) {
            Nd4j.getRandom().setSeed(seed);
            // Load the pre-trained model if the model exist
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);

            // test parking lot occupancy with test dataset
            testWithTestDataset(testIter);

            // test parking lot occupancy with webcam
            testWithWebcam();
        } else {
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorBoxes);
            //If no pre-trained model exist
            log.info("Build model...");
            // Load pre-trained Tiny-YOLO model
            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();
            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = getComputationGraph(pretrained, priors, fineTuneConf);
            System.out.println(model.summary(
                    InputType.convolutional(
                            org.project.group2.CarIterator.yoloHeight,
                            org.project.group2.CarIterator.yoloWidth,
                            nClasses)
                    )
            );

            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            // Set the listener
            model.setListeners(new ScoreIterationListener(10), new StatsListener(storage));

            for (int i = 1; i < nEpochs + 1; i++) {
                trainIter.reset();

                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }

                log.info("*** Completed epoch {} ***", i);
            }

            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
            System.out.println("Re-run the program for testing the object detection");
            System.exit(0);
        }


    }

    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {
        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("leaky_re_lu_7")
                .removeVertexKeepConnections("conv2d_9")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_9",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_8")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_9")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }

    private static void testWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
        System.out.println("Press any key to go through the test dataset.\nPress Esc key to exit the program.");

        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Parking Lot Occupancy Detection With Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()) {
            DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            YoloUtils.nms(objs, 0.4);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));
            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
            canvas.showImage(converter.convert(convertedMat_big));
            checkKeyPressed(canvas, 0);
        }

        canvas.dispose();
    }

    private static void testWithWebcam() throws InterruptedException {
        System.out.println("Webcam starting.\nPress Esc to exit the program.");

        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(
                org.project.group2.CarIterator.yoloHeight,
                org.project.group2.CarIterator.yoloWidth,
                3,
                new ColorConversionTransform(COLOR_BGR2RGB)
        );
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        FrameGrabber grabber = null;

        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            assert grabber != null;
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("Parking Lot Occupancy Detection With Webcam");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

        while (true) {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            if (thread == null) {
                thread = new Thread(() -> {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(org.project.group2.CarIterator.yoloWidth, org.project.group2.CarIterator.yoloHeight));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
                            YoloUtils.nms(objs, 0.4);
                            rawImage = drawResults(objs, rawImage, w, h);
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            checkKeyPressed(canvas, 30);
        }
    }

    private static void checkKeyPressed(CanvasFrame canvas, int delay) throws InterruptedException {
        KeyEvent t = null;

        try {
            t = canvas.waitKey(delay);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        if (t != null) {
            if (t.getKeyCode() == KeyEvent.VK_ESCAPE) {
                // when Esc key is pressed, hide canvas, dispose the canvas, then exit the program
                canvas.setVisible(false);
                canvas.dispose();
                System.exit(0);
            }
        }
    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / 13);
            int y1 = (int) Math.round(h * xy1[1] / 13);
            int x2 = (int) Math.round(w * xy2[0] / 13);
            int y2 = (int) Math.round(h * xy2[1] / 13);
            // draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            // display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(
                    mat,
                    new Point(x1 + 2, y2 - 2),
                    new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)),
                    colormap[obj.getPredictedClass()],
                    FILLED,
                    0,
                    0
            );
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
        }

        return mat;
    }
}