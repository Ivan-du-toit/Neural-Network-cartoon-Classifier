package pictureclassifier;

import Custom.Reporting.XML.XMLReport;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

/**
 *
 * @author Ivan du Toit <s29363412>
 */
public class PictureClassifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            if (hasParam(args, "-h") && hasParam(args, "-help")) {
                printHelp();
                System.exit(0);
            }
            float learningRate = getFloatParamValue(args, "-l", 0.1f);
            float momentum = getFloatParamValue(args, "-m", 0.8f);
            int numberOfHiddenNodes = getIntParamValue(args, "-n", 30);
            int numberOfEpochs = getIntParamValue(args, "-e", 1000);
            int runs = getIntParamValue(args, "-r", 30);;
            
                    
            File cartoonDir = new File("Cartoons/");
            File photoDir = new File("Photos/");
            
            File[] photoList = photoDir.listFiles();
            File[] cartoonList = cartoonDir.listFiles();
            DataElement[] elements = new DataElement[cartoonList.length + photoList.length];
            
            int counter = 0;
            for (; counter<photoList.length; ++counter) {
                 elements[counter] = new DataElement(photoList[counter], true);
            }
            for (int k=0; k<cartoonList.length; ++k) {
                 elements[counter + k] = new DataElement(cartoonList[k], false);
            }
            
            DateFormat dateFormat = new SimpleDateFormat("dd-MM-yyyy HH-mm-ss");
            Date date = new Date();
            XMLReport report = new XMLReport(dateFormat.format(date) + ".xml");
            report.addParam("learningRate", String.valueOf(learningRate));
            report.addParam("momentum", String.valueOf(momentum));
            report.addParam("numberOfHiddenUnits", String.valueOf(numberOfHiddenNodes));
            report.addParam("maxEpochs", String.valueOf(numberOfEpochs));
            report.addParam("runs", String.valueOf(runs));
            DataElement[] trainingSet = new DataElement[(int) (elements.length*0.8)];
            DataElement[] validationSet = new DataElement[elements.length-trainingSet.length];
            DataElement[] bkelements = elements;
            bkelements = Arrays.copyOf(elements, elements.length);
            for (int r=0; r<runs; ++r) {
                elements = Arrays.copyOf(bkelements, bkelements.length);
                System.out.println("Starting run: " + (r+1));
                report.startRun(r);            
                
                int elementsLeft = elements.length;
                int index = 0;
                Random rand = new Random();
                for (int n=0; n<trainingSet.length; ++n) {
                    index = rand.nextInt(elementsLeft-1);
                    trainingSet[n] = elements[index];
                    elements[index] = elements[--elementsLeft];
                }
                for (int h=0; h<validationSet.length; ++h) {
                    if (elementsLeft != 1)
                        index = rand.nextInt(elementsLeft-1);
                    else
                        index = 0;
                    validationSet[h] = elements[index];
                    elements[index] = elements[--elementsLeft];
                }
                NNetwork NN = new NNetwork(learningRate, momentum, numberOfHiddenNodes, DataElement.numberOfInputs, report);
                NN.train(trainingSet, validationSet, numberOfEpochs);
                
                float genAccuracy = 0;
                for (int t=0; t<validationSet.length; ++t) {
                    genAccuracy += NN.classify(validationSet[t]);
                }
                genAccuracy = (genAccuracy/(float)validationSet.length)*100;
                report.runResult(genAccuracy);
                
                report.save();
            }
            
            report.save();
            
        } catch (IOException ex) {
            System.out.println("File I/O Problem");
        }
    }
    
    public static boolean hasParam(String[] params, String paramName) {
        for (int k=0; k<params.length; ++k) {
            if (params[k].equals(paramName))
                return true;
        }
        return false;
    }
    
    public static float getFloatParamValue(String[] params, String paramName, float defaultValue) {
        if (hasParam(params, paramName))
            return Float.valueOf(getParamValue(params, paramName, ""));
        return defaultValue;
    }
    
    public static int getIntParamValue(String[] params, String paramName, int defaultValue) {
        if (hasParam(params, paramName))
            return Integer.valueOf(getParamValue(params, paramName, ""));
        return defaultValue;
    }
    
    public static String getParamValue(String[] params, String paramName, String defaultValue) {
        for (int k=0; k<params.length-1; ++k) {
            if (params[k].equals(paramName))
                return params[k+1];
        }
        return defaultValue;
    }
    
    public static void printHelp() {
        System.out.println("This program expects the training data to be in 2 directories in the current working dir called \"Cartoons\" and \"Photos\".");
        System.out.println("This program supports the following parameters:");
        System.out.println("[-h or -help or ?]: prints this help screen");
        System.out.println("[-l]: Specifies the learning rate to be used. Default value: 0.1");
        System.out.println("[-m]: Specifies the momentum of the back propagation function. Default value: 0.8");
        System.out.println("[-n]: Specifies the number of hidden units in the network. Default value: 30");
        System.out.println("[-e]: The number of epochs that the network should train for.");
        System.out.println("[-r]: The number of runs that the network perform with the same parameters but with random training sets.");
        System.exit(0);
    }
}
