import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.SimpleConfiguration;

import java.io.File;

public class OWLReasoningExample {
    public static void main(String[] args) {
        try {
            // Load the ontology
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            File file = new File("/path/to/your/ontology.owl");
            OWLOntology ontology = manager.loadOntologyFromOntologyDocument(file);
            System.out.println("Loaded ontology: " + ontology.getOntologyID());

            // Create a reasoner
            OWLReasonerFactory reasonerFactory = new Reasoner.ReasonerFactory();
            OWLReasoner reasoner = reasonerFactory.createReasoner(ontology, new SimpleConfiguration());

            // Perform reasoning
            reasoner.precomputeInferences();
            System.out.println("Reasoning completed");

            // Save the inferred ontology
            File outputFile = new File("/path/to/save/inferred_ontology.owl");
            manager.saveOntology(ontology, IRI.create(outputFile.toURI()));
            System.out.println("Inferred ontology saved to: " + outputFile.getAbsolutePath());

            // Dispose the reasoner
            reasoner.dispose();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}