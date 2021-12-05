package org.project.group2;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CarIterator {
    private static final Logger log = LoggerFactory.getLogger(CarIterator.class);

    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static Path trainDir, testDir;
    private static FileSplit trainData, testData;
    private static ImageTransform transform;

    // parameters to match the pretrained TinyYOLO model
    public static final int yoloHeight = 416;
    public static final int yoloWidth = 416;
    private static final int nChannels = 3;
    public static final int gridHeight = 13;
    public static final int gridWidth = 13;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, Path dir, int batchSize, boolean isTraining) throws Exception {
        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(
                yoloHeight,
                yoloWidth,
                nChannels,
                gridHeight,
                gridWidth,
                new VocLabelProvider(dir.toString())
        );

        if (isTraining && transform != null) {
            recordReader.initialize(split, transform);
        } else {
            recordReader.initialize(split);
        }

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, trainDir, batchSize, true);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, testDir, batchSize, false);
    }

    public static void setup() throws IOException {
        log.info("Load data...");
        loadData();
        log.info("Transforming image...");
        transform = imageTransform();
        trainDir = Paths.get(dataDir, "train");
        testDir = Paths.get(dataDir, "test");
        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
    }

    private static void loadData() throws IOException {
        dataDir = new ClassPathResource("cars").getFile().getPath();
    }

    private static ImageTransform imageTransform() {
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(rng, 15);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage, 0.3)
        );

        return new PipelineImageTransform(pipeline, false);
    }
}
