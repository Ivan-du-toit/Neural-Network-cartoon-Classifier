package pictureclassifier;

import Custom.Reporting.XML.XMLReport;
import java.util.Random;

/**
 *
 * @author Ivan du Toit <s29363412>
 */
public class NNetwork {
    private float learningRate;
    private float momentum;
    private int numberOfHiddenNodes;
    private int numberOfInputs;
    //private float trainingAccuracy;
    
    private final short numberOfLayers = 2;
    private final short numberOfOutputs = 2;
    private double[][][] weights;
    private double[][][] deltaW;
    
    private XMLReport report;

    public NNetwork(float learningRate, float momentum, int numberOfHiddenNodes, int numberOfInputs, XMLReport report) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.numberOfHiddenNodes = numberOfHiddenNodes;
        this.numberOfInputs = numberOfInputs;
        this.report = report;
        //initWeights();
    }
    
    private void initWeights() {
        weights = new double[numberOfLayers][][];
        deltaW = new double[numberOfLayers][][];
        Random rand = new Random(23498723498712390l);
        double faninWeight = 1/Math.sqrt(DataElement.numberOfInputs);
        weights[0] = new double[numberOfHiddenNodes][numberOfInputs];
        deltaW[0] = new double[numberOfHiddenNodes][numberOfInputs];
        for (int j=0; j<numberOfHiddenNodes; ++j) {
            for (int i=0; i<numberOfInputs; ++i) {
                weights[0][j][i] = (faninWeight*rand.nextGaussian());
                deltaW[0][j][i] = 0;
            }
        }
        
        faninWeight = 1/Math.sqrt(numberOfHiddenNodes+1);
        weights[1] = new double[numberOfOutputs][numberOfHiddenNodes+1];
        deltaW[1] = new double[numberOfOutputs][numberOfHiddenNodes+1];
        for (int j=0; j<numberOfHiddenNodes+1; ++j) {
            for (int k=0; k<numberOfOutputs; ++k) {
                weights[1][k][j] = (faninWeight*rand.nextGaussian());
                deltaW[1][k][j] = 0;
            }
        }
    }
    
    public void train(DataElement[] trainingSet, DataElement[] validationSet, int maxEpoc) {
        initWeights();
        int epoch = 0;
        double trainingAccuracy;
        
        double[] hiddenValues = new double[numberOfHiddenNodes+1];
        //Add the bias unit
        hiddenValues[numberOfHiddenNodes] = -1;
        double[] inputs;
        double[] net = new double[numberOfHiddenNodes];
        double[] output = new double[numberOfOutputs];
        double[] actual = new double[2];
        boolean[] target;
        double[] deltaO = new double[numberOfOutputs];
        double[] deltaY = new double[numberOfHiddenNodes];
        
        for (int j=0; j<numberOfHiddenNodes; ++j) {
            deltaY[j] = 0;
        }
        
        while (epoch < maxEpoc) {
            trainingAccuracy = 0;
            epoch++;

            //Go through each of the training samples
            for (DataElement trainingExample : trainingSet) {
                inputs = trainingExample.getData();
                
                //Compute the net(y) to each hidden unit
                for (int j=0; j<numberOfHiddenNodes; ++j) {
                    net[j] = 0;
                    for (int i=0; i<numberOfInputs; ++i) {
                        net[j] += weights[0][j][i]*inputs[i];
                    }
                }
                
                //Compute the activation y(j) of each hidden unit
                for (int j=0; j<numberOfHiddenNodes; ++j) {
                    hiddenValues[j] = 1/(1+Math.pow(Math.E, -net[j]));
                }
                
                //Compute the net input, netok , to each output unit.
                for (int k=0; k<numberOfOutputs; ++k) {
                    net[k] = 0;
                    for (int j=0; j<numberOfHiddenNodes+1; ++j) {
                        net[k] += weights[1][k][j]*hiddenValues[j];
                    }
                }
                
                target = trainingExample.getClassification();
                //Compute the activation O(k) of each output unit
                for (int k=0; k<numberOfOutputs; ++k) {
                    output[k] = 1/(1+Math.pow(Math.E, -net[k]));
                    
                    //Determine if the activation of a(k) should be 0 or 1:
                    //O(k) >= 0.7 then a(k) = 1
                    //O(k) <= 0.3 then a(k) = 0
                    if (output[k] >= 0.7)
                        actual[k] = 1;
                    else if (output[k] <= 0.3)
                        actual[k] = 0;
                    else {
                        //Else record a classification error for this training pattern
                        //I don't know if this is correct but it seems reasonable
                        actual[k] = output[k];
                    }
                }
                
                //Determin if the target output has been cortectly predicted
                if (((output[0] >= 0.7 && target[0]) || (output[0] <= 0.3  && !target[0])) &&
                    ((output[1] >= 0.7 && target[1]) || (output[1] <= 0.3  && !target[1])))
                    trainingAccuracy++;
                //Calculate the error signal for each output
                for (int k=0; k<numberOfOutputs; ++k) {
                    deltaO[k] = -(((target[k])?1.0:0.0) - output[k])*(1.0-output[k])*output[k];
                }
                
                
                //Calculate the new weights values for the hidden to output weights
                for (int k=0; k<numberOfOutputs; ++k) {
                    for (int j=0; j<numberOfHiddenNodes; ++j) {
                        deltaW[1][k][j] = -learningRate*deltaO[k]*hiddenValues[j] + momentum*deltaW[1][k][j];
                        weights[1][k][j] += deltaW[1][k][j];
                    }
                 }
                
                //Calculate the error signal for each hidden unit:
                for (int j=0; j<numberOfHiddenNodes; ++j) {
                    deltaY[j] = 0;
                    for (int k=0; k<numberOfOutputs; ++k) {
                        deltaY[j] += deltaO[k]*weights[1][k][j]*(1-hiddenValues[j])*hiddenValues[j];
                        //deltaY[j] += deltaO[k]*weights[0][k][j]*(1-hiddenValues[j])*hiddenValues[j];
                    }
                }
                
                //Calculate the new weight values for the weights between hidden neuron j and input neuron i
                // deltaV(J(i)) = -learningRate*deltaY(j)*Z(i) + alpha*deltaV(J(i)) //That last part is important
                // V(J(i)) = deltaV(J(i))
                 for (int j=0; j<numberOfHiddenNodes; ++j) {
                    for (int i=0; i<numberOfInputs; ++i) {
                        deltaW[0][j][i] = -learningRate*deltaY[j]*inputs[i] + momentum*deltaW[0][j][i];
                        weights[0][j][i] += deltaW[0][j][i];
                    }
                }
            }

            trainingAccuracy = (trainingAccuracy/(float)trainingSet.length)*100;
            float genAccuracy = 0;
            for (int t=0; t<validationSet.length; ++t) {
                genAccuracy += classify(validationSet[t]);
            }
            genAccuracy = (genAccuracy/(float)validationSet.length)*100;
            report.addEpoch(epoch, trainingAccuracy, genAccuracy);
            //System.out.println("Epoc: " + epoch + " Test Accuracy: " + trainingAccuracy + " gen accuracy: " + genAccuracy);
        }
    }
    
    public int classify(DataElement example) {
        double[] inputs = example.getData();
        double[] net = new double[numberOfHiddenNodes];
        double[] hiddenValues = new double[numberOfHiddenNodes+1];
        double[] output = new double[numberOfOutputs];
        boolean[] target;
        
        //Compute the net(y) to each hidden unit
        for (int j=0; j<numberOfHiddenNodes; ++j) {
            net[j] = 0;
            for (int i=0; i<numberOfInputs; ++i) {
                net[j] += weights[0][j][i]*inputs[i];
            }
        }

        //Compute the activation y(j) of each hidden unit
        for (int j=0; j<numberOfHiddenNodes; ++j) {
            hiddenValues[j] = 1/(1+Math.pow(Math.E, -net[j]));
        }

        //Compute the net input, netok , to each output unit.
        for (int k=0; k<numberOfOutputs; ++k) {
            net[k] = 0;
            for (int j=0; j<numberOfHiddenNodes+1; ++j) {
                net[k] += weights[1][k][j]*hiddenValues[j];
            }
        }
        
        
        //Compute the activation O(k) of each output unit
        for (int k=0; k<numberOfOutputs; ++k) {
            output[k] = 1/(1+Math.pow(Math.E, -net[k]));
        }
        
        target = example.getClassification();
        if (((output[0] >= 0.7 && target[0]) || (output[0] <= 0.3  && !target[0])) &&
                ((output[1] >= 0.7 && target[1]) || (output[1] <= 0.3  && !target[1])))
                return 1;
        return 0;
    }
}
