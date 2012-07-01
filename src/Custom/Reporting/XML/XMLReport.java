package Custom.Reporting.XML;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * This class outputs the results from the algorithms to an xml file for later analysis 
 * @author Ivan du Toit <s29363412>
 */
public class XMLReport {
    private Element root;
    private Document doc;
    private String fileName;
    private Element paramRoot;
    private Element epochRoot;
    
    public XMLReport(String fileName) {
        this.fileName = fileName;
        try {
            //Create instance of DocumentBuilderFactory
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            //Get the DocumentBuilder
            DocumentBuilder docBuilder = factory.newDocumentBuilder();
            //Create blank DOM Document
            doc = docBuilder.newDocument();
            //create the root element
            root = doc.createElement("report");
            doc.appendChild(root);
            
            paramRoot = doc.createElement("parameters");
            epochRoot = doc.createElement("run");
            root.appendChild(paramRoot);
            root.appendChild(epochRoot);
            
        } catch (ParserConfigurationException e) {
            System.out.println("Could not parse file");
        }
    }
    
    public void addParam(String name, String value) {
        Element paramNode = doc.createElement("param");
        paramNode.setAttribute(name, value);
        paramRoot.appendChild(paramNode);
    }
    
    public void addEpoch(int epoch, double testAccuracy, double genAccuracy) {
        Element EntryNode = doc.createElement("epoch");
        
        EntryNode.setAttribute("epochNumber", String.valueOf(epoch));
        EntryNode.setAttribute("testAccuracy", String.valueOf(testAccuracy));
        EntryNode.setAttribute("generalAccuracy", String.valueOf(genAccuracy));
        
        epochRoot.appendChild(EntryNode);
    }
    
    public void runResult(double genAccuracy) {
        epochRoot.setAttribute("runResult", String.valueOf(genAccuracy));
    }
    
    public void startRun(int runNumber) {
        epochRoot = doc.createElement("run");
        root.appendChild(epochRoot);
        epochRoot.setAttribute("runNumber", String.valueOf(runNumber));
    }
    
    public void save() {
        try {
            TransformerFactory tranFactory = TransformerFactory.newInstance(); 
            Transformer aTransformer = tranFactory.newTransformer(); 

            Source src = new DOMSource(doc); 
            Result dest = new StreamResult(fileName); 
            aTransformer.transform(src, dest);
        } catch (TransformerConfigurationException ex) {
            System.out.println("Could not save to file(conf)");
        } catch (TransformerException ex) {
            System.out.println("Could not save to file");
        }
    }
}
