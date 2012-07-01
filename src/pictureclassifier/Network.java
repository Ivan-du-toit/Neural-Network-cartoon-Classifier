package pictureclassifier;

import java.util.Random;

/**
 *
 * @author Ivan du Toit <s29363412>
 */
public class Network {
    private float learningRate;
    private float momentum;
    private int numberOfHiddenNodes;
    private float trainingAccuracy;
    private double[][] weights;

    public Network(float learningRate, float momentum, int numberOfHiddenNodes) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.numberOfHiddenNodes = numberOfHiddenNodes;
        
        initWeights();
        
    }
    
    private void initWeights() {
        //Init the weights
        weights = new double[2][];
        weights[0] = new double[(DataElement.numberOfInputs+1)*numberOfHiddenNodes];
        Random rand = new Random();
        double faninWeight = 1/Math.sqrt(DataElement.numberOfInputs+1);
        for (int w=0; w<weights[0].length; ++w) {
            weights[0][w] = (faninWeight*rand.nextGaussian());
        }
        
        weights[1] = new double[numberOfHiddenNodes*2+2];
        for (int w=0; w<weights[1].length; ++w) {
            weights[1][w] = (faninWeight*rand.nextGaussian());
        }
    }
    
    public boolean[] train(DataElement[] trainingSet, DataElement[] testSet, int maxEpoch) {
        int epoch = 0;
        double net = 0;
        double[] inputs;
        
        //Also add the bias unit
        double[] hiddenOutputs = new double[numberOfHiddenNodes+1];
        //Set the of set bias unit
        hiddenOutputs[numberOfHiddenNodes] = -1;
        
        double[] output = new double[2];
        double[] accuracy = new double[2];
        boolean[] actual;
        double[] deltaO = new double[]{0, 0};
        //Also add the bias unit
        double[] errors = new double[numberOfHiddenNodes+1];
        
        
        while (epoch < maxEpoch) {
            trainingAccuracy = 0;
            ++epoch;
            for (int e=0; e<trainingSet.length; ++e) {
                //Gets all the input values including the bias unit
                inputs = trainingSet[e].getData();
                for (int h=0; h<numberOfHiddenNodes; ++h) {
                    net = 0;
                    for (int j=0; j<inputs.length; ++j) {
                        net += inputs[j]*weights[0][inputs.length*h+j];
                    }
                    hiddenOutputs[h] = (double)1 / (1+Math.pow(Math.E, net));
                }
                
                actual = trainingSet[e].getClassification();
                for (int o=0; o<2; ++o) {
                    for (int h=0; h<hiddenOutputs.length; ++h) {
                        net += hiddenOutputs[h]*weights[1][hiddenOutputs.length*o+h];
                    }
                    output[o] = 1 / (1+Math.pow(Math.E, -net));
                    net = 0;
                    if (output[o] >= 0.7) {
                        if (actual[o] == true)
                            accuracy[o] = 1;
                        else
                            accuracy[o] = 0;
                    }
                    else if (output[o] <= 0.3) {
                        if (actual[o] == false)
                            accuracy[o] = 1;
                        else
                            accuracy[o] = 0;
                    }
                    else
                        accuracy[o] = output[o];
                }
                
                //Calculate the error signal for each output
                if (accuracy[0]+accuracy[1] ==  2)
                    trainingAccuracy += 1;
                else { 
                    deltaO[0] = -(((actual[0])?1.0:0.0) - output[0])*(1-output[0])*output[0];
                    deltaO[1] = -(((actual[1])?1.0:0.0) - output[1])*(1-output[1])*output[1];
                }
                
                double[][] oldWeights = new double[weights.length][];
                for (int w=0; w<oldWeights.length; ++w) {
                    oldWeights[w] = new double[weights[w].length];
                    for (int v=0; v<oldWeights[w].length; ++v) 
                        oldWeights[w][v] = weights[w][v];
                }
                
                //update hidden to output weights
                for (int k=0; k<2; ++k) {
                    for (int j=0; j<hiddenOutputs.length; ++j) {
                        weights[1][k*hiddenOutputs.length + j] += -learningRate*deltaO[k]*hiddenOutputs[j] + momentum*weights[1][k*hiddenOutputs.length + j];
                    }
                }
                
                //Calculate the error signal for each hidden unit.
                for (int j=0; j<hiddenOutputs.length; ++j) {
                    errors[j] = 0; 
                    for (int k=0; k<2; ++k) {
                        errors[j] += deltaO[k]*oldWeights[1][k*hiddenOutputs.length + j]*(1-hiddenOutputs[j])*hiddenOutputs[j];
                    }
                }
                
                for (int j=0; j<hiddenOutputs.length-1; ++j) {
                    for (int i=0; i<inputs.length; ++i) {
                        weights[0][j*inputs.length+i] += -learningRate*errors[j]*inputs[i] + momentum*weights[0][j*inputs.length+i];
                    }
                }
            }
            
            System.out.println("#Corect: " + trainingAccuracy);
            trainingAccuracy = (trainingAccuracy/(float)trainingSet.length)*100;
            float genAccuracy = 0;
            for (int t=0; t<testSet.length; ++t) {
                genAccuracy += classify(testSet[t]);
            }
            genAccuracy = (genAccuracy/(float)testSet.length)*100;
            System.out.println("Epoc: " + epoch + " Test Accuracy: " + trainingAccuracy + " gen accuracy: " + genAccuracy);
        }
        
        return new boolean[]{false, false};
    }
    
    private int classify(DataElement test) {
        //TODO implement this
        return 0;
    }
}
