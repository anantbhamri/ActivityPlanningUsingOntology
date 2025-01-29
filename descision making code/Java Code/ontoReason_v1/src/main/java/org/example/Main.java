package org.example;
//package org.mindswap.pellet.examples;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.util.InferredAxiomGenerator;
import org.semanticweb.owlapi.util.InferredOntologyGenerator;

import com.clarkparsia.pellet.owlapiv3.PelletReasoner;
import com.clarkparsia.pellet.owlapiv3.PelletReasonerFactory;
//import openllet.owlapi.OpenlletReasonerFactory;

import java.io.File;
import java.util.Set;
import java.util.ArrayList;
import java.util.List;
//import org.semanticweb.HermiT.Reasoner.ReasonerFactory;


import org.semanticweb.owlapi.util.InferredClassAssertionAxiomGenerator;


public class Main {
    public static void main(String[] args) {

        try {
            // Load the ontology
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            File file = new File("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_NuclearPlantmultiVehicleV2.owl");
            OWLOntology ontology = manager.loadOntologyFromOntologyDocument(file);
            System.out.println("Loaded ontology: " + ontology.getOntologyID());


            PelletReasoner reasoner = PelletReasonerFactory.getInstance().createReasoner( ontology );
            System.out.println("done.");

            reasoner.getKB().realize();
            reasoner.getKB().printClassTree();



//
//            // Query the reasoned ontology
//            OWLDataFactory dataFactory = manager.getOWLDataFactory();
//
//            // Print all individuals in the ontology
//            ontology.individualsInSignature().forEach(individual -> {
//                System.out.println("Individual: " + individual.getIRI());
//            });
//
//            // Get the base IRI of the ontology
//            IRI baseIRI = ontology.getOntologyID().getOntologyIRI().orElseThrow(() -> new RuntimeException("Ontology IRI not found"));
//
//            // Construct the IRI for st_CPO
//            IRI stCPOIRI = IRI.create(baseIRI + "#st_CPO");
//            OWLNamedIndividual stCPO = dataFactory.getOWLNamedIndividual(stCPOIRI);
//
//            // Check if the individual stCPO exists
//            if (!ontology.containsIndividualInSignature(stCPO.getIRI())) {
//                System.out.println("Individual st_CPO not found in the ontology.");
//                return;
//            }
//
//            // Print all object properties of stCPO
//            System.out.println("Object properties of stCPO:");
//            ontology.objectPropertyAssertionAxioms(stCPO).forEach(axiom -> {
//                OWLObjectPropertyExpression property = axiom.getProperty();
//                OWLIndividual target = axiom.getObject();
//                System.out.println("stCPO has property " + property.asOWLObjectProperty().getIRI() + " to: " + target.asOWLNamedIndividual().getIRI());
//            });
//
//            // Find and print the class of the individual st_CPO
//            System.out.println("Classes of st_CPO:");
//            ontology.classAssertionAxioms(stCPO).forEach(axiom -> {
//                OWLClassExpression classExpression = axiom.getClassExpression();
//                System.out.println("st_CPO is an instance of class: " + classExpression.asOWLClass().getIRI());
//            });
//
//            // Create a reasoner
//            OWLReasonerFactory reasonerFactory = new OpenlletReasonerFactory();
////            OWLReasonerFactory reasonerFactory = new ReasonerFactory();
//            OWLReasoner reasoner = reasonerFactory.createReasoner(ontology);
//
//            // Perform reasoning
//            reasoner.precomputeInferences();
//            System.out.println("Reasoning completed");
//
//            List<InferredAxiomGenerator<? extends OWLAxiom>> generators = new ArrayList<>();
//            generators.add(new InferredClassAssertionAxiomGenerator());
//            InferredOntologyGenerator iog = new InferredOntologyGenerator(reasoner, generators);
//            iog.fillOntology(manager.getOWLDataFactory(), ontology);
//
//
//
//            // Print all object properties of stCPO
//            System.out.println("Object properties of stCPO:");
//            ontology.objectPropertyAssertionAxioms(stCPO).forEach(axiom -> {
//                OWLObjectPropertyExpression property = axiom.getProperty();
//                OWLIndividual target = axiom.getObject();
//                System.out.println("stCPO has property " + property.asOWLObjectProperty().getIRI() + " to: " + target.asOWLNamedIndividual().getIRI());
//            });
//
//            // Find and print the class of the individual st_CPO
//            System.out.println("Classes of st_CPO:");
//            ontology.classAssertionAxioms(stCPO).forEach(axiom -> {
//                OWLClassExpression classExpression = axiom.getClassExpression();
//                System.out.println("st_CPO is an instance of class: " + classExpression.asOWLClass().getIRI());
//            });
//
//
//
//
//
//
//
//
//
//            // Save the inferred ontology
//            File outputFile = new File("/Users/bhamri/Desktop/Ontology/owlFiles/inferred_ontology.owl");
//            manager.saveOntology(ontology, IRI.create(outputFile.toURI()));
//            System.out.println("Inferred ontology saved to: " + outputFile.getAbsolutePath());

            // Dispose the reasoner
            reasoner.dispose();
        } catch (OWLOntologyCreationException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}


