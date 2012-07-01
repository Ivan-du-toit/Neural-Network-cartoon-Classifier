package pictureclassifier;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import javax.imageio.ImageIO;

/**
 *
 * @author Ivan du Toit <s29363412>
 */
public class DataElement {
    private BufferedImage img;
    private boolean[] classification;
    private double[] data;
    private final int bins = 3;
    //3 bins of size 3 and 1 bias unit + colour count
    public static final int numberOfInputs = 3*3*3+2;

    public DataElement(File file, boolean picture) throws IOException {
        img = ImageIO.read(file);
        if (picture)
            classification = new boolean[] {true, false};
        else
            classification = new boolean[] {false, true};
    }

    public boolean[] getClassification() {
        return classification;
    }
    
    public double[] getData() {
        if (data == null) {
            data = new double[numberOfInputs];
            int height = img.getHeight();
            int width = img.getWidth();
            
            //Crazy idea count the number of colours in the image.
            HashMap<Integer, Boolean> colours = new HashMap<>();
            
            for(int i = 0; i < height; i++){
                for(int j = 0; j < width; j++){
                    int argb = img.getRGB(j, i);
                    if (!colours.containsKey(argb))
                        colours.put(argb, Boolean.TRUE);
                    int rgb[] = new int[] {
                        (argb >> 16) & 0xff, //red
                        (argb >>  8) & 0xff, //green
                        (argb      ) & 0xff  //blue
                    };
                    //System.out.println("rgb: " + rgb[0] + " " + rgb[1] + " " + rgb[2]);
                    //redMean = redMean.add(rgb[0]);
                    
                    //Now set the correct bin in the data
                    int index = 0;
                    if (rgb[0] > 85) {
                        if (rgb[0] > 170)
                            index = (data.length/bins)*2;
                        else
                            index = data.length/bins;
                    }
                    if (rgb[1] > 85) {
                        if (rgb[1] > 170)
                            index += ((data.length/bins)/bins)*2;
                        else
                            index += (data.length/bins)/bins;
                    }
                    if (rgb[2] > 85) {
                        if (rgb[2] > 170)
                            index += (((data.length/bins)/bins)/bins)*2;
                        else
                            index += ((data.length/bins)/bins)/bins;
                    }
                    data[index]++;
                }
            }
            
            //normalize the data
            int numberOfPixels = width*height;
            for (int k=0; k<data.length-2; ++k) {
                data[k] /= numberOfPixels;
            }
            /*if (classification[0])
                data[data.length-2] = 1;
            else
                data[data.length-2] = 0;*/
            double numColours = (double)colours.size();
            data[data.length-2] = numColours/numberOfPixels;
            data[data.length-1] = -1;
        }
        return data;
    }
}
