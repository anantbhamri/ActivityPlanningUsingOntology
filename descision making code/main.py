import os
from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt
import ollama
import sys
import time
import plotly.graph_objects as go
import itertools

def get_corresponding_iri(ontology, name):
    iriToSearch = ontology.base_iri + name 
    allMatches = ontology.search(iri=iriToSearch)
    if len(allMatches) == 0:
        print("Did not find any match for: ", name)
        return
    elif len(allMatches) == 1:
        return allMatches[0]
    else:
        for match in allMatches:
            if match.name == name:
                return match


# onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1.owl").load()
onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_multiVehicle.owl").load()
# onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_NuclearPlantmultiVehicle.owl").load()
# onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/approach1_reasonedOntology.owl").load()

fileSaveName = "/Users/bhamri/Desktop/Ontology/owlFiles/saved_ontologyV2.owl"
reason = True
if reason:
    # Measure the time taken for reasoning
    start_time = time.time()
    # Perform reasoning
    with onto:
        sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True)
        print("Reasoning completed")
    end_time = time.time()
    reasoning_time = end_time - start_time
    print(f"Reasoning took {reasoning_time:.2f} seconds")

listOfActionsOfInterest = ["goEAction", "goWAction", "goSAction", "goNAction",
        "closeBoundaryAction", "openBoundaryAction",
        "closeLockAction", "openLockAction",
        "cs_closeBoundaryActionPlease", "cs_openBoundaryActionPlease",
        "cs_closeLockActionPlease", "cs_openLockActionPlease",
        "cs_goWActionPleasePlease", "cs_goEActionPleasePlease", "cs_goSActionPlease", "cs_goNActionPlease", "cs_callAnotherAgentAction",
        ]

baseIRIOnto = onto.base_iri
# Interested in states for the knowledge graph
agentStateDef = get_corresponding_iri(onto, "agentState")
compositeStateDef = get_corresponding_iri(onto, "compositeState")
coupledWithDef = get_corresponding_iri(onto, "coupledWith")
downstreamCouplingDef = get_corresponding_iri(onto, "downstreamCoupling")
sameStateAgentDef = get_corresponding_iri(onto, "sameStateAgent")
agentClass = get_corresponding_iri(onto, "agentClass")
motionAgentStateDef = get_corresponding_iri(onto, "motionAgentState")

ontoDef12 = get_corresponding_iri(onto, "cs_clearRubbleActionPlease")

# Interested in relationships between individuals (Nominal Actions and coupling)
actionsOfInterest = {
    get_corresponding_iri(onto, action) for action in [
    # onto.search_one(iri=f"*{action}") for action in [
        "goEAction", "goWAction", "goSAction", "goNAction",
        "closeBoundaryAction", "openBoundaryAction",
        "closeLockAction", "openLockAction",
        "clearRubbleAction",
        "powerCutEventReach", 
        "turnOffGeneratorAction", "turnOnGeneratorAction",
        "turnOffPumpAction", "turnOnPumpAction", 
        "timePassageAction",
        "closeVentAction", "openVentAction",

        "cs_closeBoundaryActionPlease", "cs_openBoundaryActionPlease",
        "cs_closeLockActionPlease", "cs_openLockActionPlease",
        "cs_goWActionPlease", "cs_goEActionPlease", "cs_goSActionPlease", "cs_goNActionPlease",
        "cs_clearRubbleActionPlease", 
        "cs_powerCutEventReachPlease", 
        "cs_turnOffGeneratorActionPlease", "cs_turnOnGeneratorActionPlease",
        "cs_turnOffPumpActionPlease", "cs_turnOnPumpActionPlease", 
        "cs_timePassageActionPlease",
        "cs_closeVentAction", "cs_openVentActionPlease",
        "cs_disconnectFuelTruckActionPlease", "cs_connectFuelTruckActionPlease",
        
        "cs_callAnotherAgentAction", "cs_noActionPlease",


    ]
}
actions = [
        "goEAction", "goWAction", "goSAction", "goNAction",
        "closeBoundaryAction", "openBoundaryAction",
        "closeLockAction", "openLockAction", 
        "clearRubbleAction",
        "powerCutEventReach", 
        "turnOffGeneratorAction", "turnOnGeneratorAction",
        "turnOffPumpAction", "turnOnPumpAction", 
        "timePassageAction",
        "closeVentAction", "openVentAction",
        "disconnectFuelTruckAction", "connectFuelTruckAction",
    ]


# Function to find both out-edges and in-edges of all nodes
def find_all_edges(G):
    out_edges = []
    in_edges = []
    for node in G.nodes():
        out_edges.extend(list(G.out_edges(node)))
        in_edges.extend(list(G.in_edges(node)))
    return out_edges, in_edges

def drawGraph(G):
    # Find all out-edges and in-edges
    out_edges, in_edges = find_all_edges(G)

    pos = nx.spring_layout(G)

    # Draw the complete graph with all nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, arrows=True, edge_color="gray")

    # Draw the out-edges separately
    nx.draw_networkx_edges(G, pos, edgelist=out_edges, edge_color="blue", connectionstyle="arc3,rad=0.2")

    # Draw the in-edges separately
    nx.draw_networkx_edges(G, pos, edgelist=in_edges, edge_color="red", connectionstyle="arc3,rad=-0.2")

    # Draw the labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    plt.title("Complete Graph with Separate In and Out Edges")
    plt.show()

def find_agent_value(state):
    stringState = "*" + state
    corrspondingAgentValuePair = onto.search_one(iri=stringState).corrospondsTo
    classPair1 = corrspondingAgentValuePair[0].is_a[0]
    if agentClass in classPair1.ancestors():
        agent = corrspondingAgentValuePair[0]
        value = corrspondingAgentValuePair[1]
    else:
        agent = corrspondingAgentValuePair[1]
        value = corrspondingAgentValuePair[0]
    return agent, value

# Function to create composite states recursively
def create_composite_states(individual, visited, mapState2CompositeState):
    # Interested in relationships between individuals (Nominal Actions and coupling)
    actions = [
        "goEAction", "goWAction", "goSAction", "goNAction",
        "closeBoundaryAction", "openBoundaryAction",
        "closeLockAction", "openLockAction"
    ]
    global noOfCompositeStates
    if individual in visited:
        return individual
    visited.add(individual)
    
    composite_states = []
    if coupledWithDef in individual.get_properties():
        for value in individual.coupledWith:
            if isinstance(value, Thing):
                composite_state = create_composite_states(value, visited, mapState2CompositeState)
                composite_states.append(composite_state)

    if composite_states:
        composingIndsList = []
        compositeStateIndvs = []
        for composite_state in composite_states:
            noOfCompositeStates += 1
            if composite_state.name not in mapState2CompositeState:
                print("Error")
                # end the code with error
                sys.exit()    

            else:
                for compState in mapState2CompositeState[composite_state.name]:
                    compositeStateIndv = onto.compositeState(f"cSt_{individual.name}_{compState}")
                    
                    bool_isGoodState = False 
                    
                    indvClass = individual.is_a[0]
                    if agentStateDef in indvClass.ancestors():
                        compositeStateIndv.hasState.append(individual)
                    elif compositeStateDef in indvClass.ancestors():
                        listOfhasState = individual.hasState
                        for state in listOfhasState:
                            compositeStateIndv.hasState.append(state)
                    
                    composingStateIndvClass = get_corresponding_iri(onto, compState).is_a[0]
                    if agentStateDef in composingStateIndvClass.ancestors():
                        if get_corresponding_iri(onto, compState).goodState:
                            bool_isGoodState = get_corresponding_iri(onto, compState).goodState[0]
                        compositeStateIndv.hasState.append(get_corresponding_iri(onto,compState))
                    elif compositeStateDef in composingStateIndvClass.ancestors():    
                        bool_isGoodState = bool_isGoodState and get_corresponding_iri(onto, compState).goodState[0]
                        listOfhasState = get_corresponding_iri(onto, compState).hasState
                        for state in listOfhasState:
                            compositeStateIndv.hasState.append(state)
                    
                    compositeStateIndv.goodState.append(bool_isGoodState)
                    G.add_node(compositeStateIndv.name, type="compositeState")
                    node_labels[compositeStateIndv.name] = compositeStateIndv.name
                    compositeStateIndvs.append(compositeStateIndv)

                    composingIndsList.append([composite_state.name, compState, compositeStateIndv.name])
                    
                    if individual.name not in mapState2CompositeState:
                        mapState2CompositeState[individual.name] = set()
                    mapState2CompositeState[individual.name].add(compositeStateIndv.name)
                
        # Define the relationships between the composite states
        for i in range(len(composingIndsList)):
            st1 = get_corresponding_iri(onto, composingIndsList[i][0])
            compSt1 = get_corresponding_iri(onto, composingIndsList[i][2])
            for j in range(len(composingIndsList)):
                if i != j:
                    st2 = get_corresponding_iri(onto, composingIndsList[j][0])
                    compSt2 = get_corresponding_iri(onto, composingIndsList[j][2])
                    csAction = False
                    if st1 != st2:
                        # i.e. two states are different 
                        # then relationship flows from current states
                        rel1 = st1
                        rel2 = st2
                        rel1_likedNextState = get_corresponding_iri(onto, composingIndsList[i][1])
                        if compositeStateDef == rel1_likedNextState.is_a[0] and not rel1_likedNextState.goodState[0]:
                            continue
                        rel2_likedNextState = get_corresponding_iri(onto, composingIndsList[j][1])
                        if compositeStateDef == rel2_likedNextState.is_a[0] and not rel2_likedNextState.goodState[0]:
                            continue
                        
                    else:
                        # i.e. two states are different
                        # then relationship flows from composite states
                        rel1 = get_corresponding_iri(onto, composingIndsList[i][1])
                        rel2 = get_corresponding_iri(onto, composingIndsList[j][1])
                        
                        csAction = True
                    
                    
                    # Now check if two states are connected by a object property as mentioned in actions
                    for action in actions:
                        actionAdd = "cs_" + action + "Please"
                        if csAction:
                            action = "cs_" + action + "Please"
                        if rel2 in getattr(rel1,action):
                            print(compSt1.name, actionAdd, compSt2.name)
                            if compSt2 not in getattr(compSt1, actionAdd):
                                getattr(compSt1, actionAdd).append(compSt2)
                                print("done")
        return compositeStateIndvs
    else:
        G.add_node(individual.name, type="individual")
        node_labels[individual.name] = individual.name
        if individual.name not in mapState2CompositeState:
            mapState2CompositeState[individual.name] = set()
        mapState2CompositeState[individual.name].add(individual.name)
        return individual

def get_hierarchy_of_state(state, mapStatesCoupling, hierarchyMap):
    if state in hierarchyMap:
        return hierarchyMap[state], hierarchyMap
    
    if state not in mapStatesCoupling:
        return -1
    hierarchy = 0
    hierarchyAllCouplings = []
    for coupled_state in mapStatesCoupling[state]:
        val, hierarchyMap = get_hierarchy_of_state(coupled_state, mapStatesCoupling, hierarchyMap)
        hierarchyAllCouplings.append(val) 
    if len(hierarchyAllCouplings) > 0:
        hierarchy = max(hierarchyAllCouplings) + 1

    hierarchyMap[state] = hierarchy
    return hierarchy, hierarchyMap

# Function to sort the map based on the length of the sets
def sort_map_by_set_length(mapStatesCoupling):
    
    hierarchyMap = {}
    for element in mapStatesCoupling:
        val, hierarchyMap = get_hierarchy_of_state(element, mapStatesCoupling, hierarchyMap)
        if val == -1:
            print("Error")
        # assign the hierarchy value to the element in the ontology
        if val > 0 :
            get_corresponding_iri(onto, element).couplingHierarchy.append(val)

    sorted_map = dict(sorted(mapStatesCoupling.items(), key=lambda item: hierarchyMap[item[0]], reverse=False))
    return sorted_map

# Function to find a node in the graph
def find_node(graph, node):
    if graph.has_node(node):
        node_data = graph.nodes[node]
        print(f"Node {node} found with attributes: {node_data}")
    else:
        print(f"Node {node} not found in the graph")

# Function to find the shortest path using Dijkstra's algorithm
def find_shortest_path(graph, start_node, end_node):
    if not graph.has_node(start_node):
        print(f"Node {start_node} not found in graph")
        return
    if not graph.has_node(end_node):
        print(f"Node {end_node} not found in graph")
        return
    try:
        path = nx.dijkstra_path(graph, start_node, end_node)
        print(f"Shortest path from {start_node} to {end_node}: {path}")

        # Get edge labels for the path
        edge_labels = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            edge_labels.append(edge_data.get('relationship', ''))
        
        print(f"Edge labels for the path: {edge_labels}")
        return path, edge_labels
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}")
        return None, None
    
# Function to find all paths between two nodes
def find_all_paths(graph, start_node, end_node):
    try:
        paths = list(nx.all_simple_paths(graph, start_node, end_node))
        if paths:
            print(f"All paths from {start_node} to {end_node}:")
            for path in paths:
                print("Path ====    ", path)

                # Get edge labels for the path
                edge_labels = []
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i + 1])
                    edge_labels.append(edge_data.get('relationship', ''))
                print(f"Edge labels for the path: {edge_labels}")


        else:
            print(f"No paths found between {start_node} and {end_node}")
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}")

# Function to find all paths between two nodes that pass through a list of nodes and have length smaller than 5
def find_paths_through_nodes(graph, source, destination, through_nodes, avoid_nodes, min_length=1, max_length=5):
    all_paths = list(nx.all_simple_paths(graph, source, destination))
    filtered_paths = []
    
    for path in all_paths:
        if (
            all(node in path for node in through_nodes) and
            not any(node in path for node in avoid_nodes) and
            min_length <= len(path) <= max_length
        ):
            filtered_paths.append(path)
    # Sort the filtered paths by their length
    sorted_paths = sorted(filtered_paths, key=len)
    
    return sorted_paths
    # return filtered_paths

def findNumberOfSameAgentStateCouplings(coupledWithStates):
    # Let's find how many agents are there in the composing states
    respectiveAgents = {}
    noOfAgents = 0
    for state in coupledWithStates:
        agent = get_corresponding_iri(onto, state).[0]
        if agent.name not in respectiveAgents:
            respectiveAgents[agent.name] = [state]
            noOfAgents += 1
        else:
            respectiveAgents[agent.name].append(state)
    return respectiveAgents, noOfAgents

def defineRelationshipsBetweenCompositeStates(allCompositeStates, allCompositeStatesParentKeyList, G):
    for i in range(len(allCompositeStates)):
        agent_i = allCompositeStates[i][1]
        composite_st1 = get_corresponding_iri(onto, allCompositeStates[i][0])
        composing_st1 = get_corresponding_iri(onto, allCompositeStates[i][2])
        actionSt1 = get_corresponding_iri(onto, allCompositeStatesParentKeyList[allCompositeStates[i][0]][0])

        for j in range(len(allCompositeStates)):
            if i != j:
                agent_j = allCompositeStates[j][1]
                composite_st2 = get_corresponding_iri(onto, allCompositeStates[j][0])
                composing_st2 = get_corresponding_iri(onto, allCompositeStates[j][2])
                actionSt2 = get_corresponding_iri(onto, allCompositeStatesParentKeyList[allCompositeStates[j][0]][0]) 
                if agent_i == "A_Rubble" and agent_j == "A_Rubble":
                    print("Debug")
                if agent_i != agent_j:
                    # Then the composite states are of different agents 
                    # hence no action reach can be defined between non-detrimental states 
                    detrimentalAgent_i = False
                    
                    if len(get_corresponding_iri(onto, agent_i).detrimentalAgent) > 0:
                        detrimentalAgent_i = get_corresponding_iri(onto, agent_i).detrimentalAgent[0]
                    
                    if detrimentalAgent_i == True:
                        if agentStateDef in composing_st1.is_a[0].ancestors():
                            # Only define goodState as true and nothing for false
                            if len(composing_st1.goodState) == 0:
                                # Means that good state is False
                                continue

                        if len(composing_st1.goodState) > 0:
                            if composing_st1.goodState[0] == False:
                                continue
                        

                    if composite_st2 not in getattr(composite_st1, "cs_noActionPlease"):
                        getattr(composite_st1, "cs_noActionPlease").append(composite_st2)
                        print("Adding ====    " , composite_st1.name, " <-> cs_noActionPlease <->", composite_st2.name)
                        # Add edge on graph
                        G.add_edge(composite_st1.name, composite_st2.name, relationship="cs_noActionPlease")
                        edge_labels[(composite_st1.name, composite_st2.name)] = "cs_noActionPlease"
                        # G.add_edge(composite_st2.name, composite_st1.name, relationship="cs_noActionPlease")
                        # edge_labels[(composite_st2.name, composite_st1.name)] = "cs_noActionPlease"
                        
                else:   
                    csAction = False
                    if agentStateDef in composing_st1.is_a[0].ancestors() and agentStateDef in composing_st2.is_a[0].ancestors():
                        # Both are states
                        rel1 = composing_st1
                        rel2 = composing_st2

                        if len(rel1.constrainingStates) > 0: 
                            hasStatesInCompositeState = composite_st1.hasState
                            constrainingStatesInComposingState = rel1.constrainingStates
                            # if any of the constraining states are in hasSates then only the action can be performed
                            if not any(state in hasStatesInCompositeState for state in constrainingStatesInComposingState):
                                continue
                    
                    elif agentStateDef in composing_st1.is_a[0].ancestors() and compositeStateDef in composing_st2.is_a[0].ancestors():
                        # st1 is state and st2 is composite state
                        rel1 = composing_st1
                        rel2 = actionSt2
                        if composing_st2.goodState[0] == False:
                            continue
                    
                    elif compositeStateDef in composing_st1.is_a[0].ancestors() and agentStateDef in composing_st2.is_a[0].ancestors():
                        # st2 is state and st1 is composite state
                        rel1 = actionSt1
                        rel2 = composing_st2
                        if composing_st1.goodState[0] == False:
                            continue

                    else:
                        # Both are composite states
                        rel1 = composing_st1
                        rel2 = composing_st2
                        csAction = True

                    # Now check if two states are connected by a object property as mentioned in actions
                    for action in actions:
                        actionAdd = "cs_" + action + "Please"
                        if csAction:
                            action = "cs_" + action + "Please"
                        if rel2 in getattr(rel1,action):
                            if composite_st2 not in getattr(composite_st1, actionAdd):
                                getattr(composite_st1, actionAdd).append(composite_st2)
                                print("Adding ====    " , composite_st1.name, " -> ",actionAdd," ->", composite_st2.name)
                                G.add_edge(composite_st1.name, composite_st2.name, relationship= actionAdd)
                                edge_labels[(composite_st1.name, composite_st2.name)] = actionAdd

def find_edges(individuals, G, actionsOfInterest):
    for individual in individuals:
        # For all states find the relevant relationships
        indvClass = individual.is_a[0]
        addEgde = False
        if individual.name == "st_c14":
            print("Debug")
        if agentStateDef in indvClass.ancestors():
            # i.e. state = only add edges for states with downstreamCoupling = False
            if individual.downstreamCoupling[0] == False:
                addEgde = True
        elif indvClass == onto.compositeState:
            addEgde = True
        if addEgde:
            for prop in individual.get_properties():
                if prop in actionsOfInterest:
                    for value in prop[individual]:
                        if isinstance(value, Thing):  # Check if the value is another individual
                            if agentStateDef in value.is_a[0].ancestors():
                                if value.downstreamCoupling[0] == False:
                                    G.add_edge(individual.name, value.name, relationship=prop.name)
                                    edge_labels[(individual.name, value.name)] = prop.name
                            else:
                                G.add_edge(individual.name, value.name, relationship=prop.name)
                                edge_labels[(individual.name, value.name)] = prop.name
                            # edge_colors[(individual.name, value.name)] = "blue"
        
    # Print some basic information about the graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

def reason_save_ontology(onto_def, fileName):
    # Ensure all individuals in the ontology are different
    all_individuals = list(onto_def.individuals())
    AllDifferent(all_individuals)

    with onto_def:
        sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True)
        print("Reasoning completed")

    # Save the ontology to a file
    onto.save(file= fileName, format="rdfxml")

# Define a color map for edge labels
color_map = {
    "goEAction": "blue",
    "goWAction": "green",
    "goSAction": "blue",
    "goNAction": "green",
    "closeBoundaryAction": "orange",
    "openBoundaryAction": "black",
    "closeLockAction": "brown",
    "openLockAction": "black",
    "nonSymmetricCouplingReach": "cyan",
    "symmetricCouplingReach": "red"
}

# Create an empty graph
G = nx.DiGraph()

allAgents = {}
for individual in onto.individuals():
    # Add the individuals which belong to the agentState as nodes
    indvClass = individual.is_a[0]
    if agentClass in indvClass.ancestors():
        if individual.name not in allAgents:    
            allAgents[individual.name] = individual.corrospondedBy
        else:
            allAgents[individual.name].extend(individual.corrospondedBy)

node_labels = {}
noOfCompositeStates = 0

# Add individuals as nodes and create composite states
listOfIndividualStates = []
mapStatesCoupling = {}

useOldCompositeStates = False

visited = set()
mapState2CompositeState = {}

# Add individual states as nodes and define their hierarchy
for individual in onto.individuals():
    # Add the individuals which are states as nodes
    indvClass = individual.is_a[0]
    if agentStateDef in indvClass.ancestors():
        # find_agent_value
        agentName, agentValue = find_agent_value(individual.name)
        individual.correspondingAgent.append(agentName)
        individual.correspondingValue.append(agentValue)

        # Define the downstreamCoupling value as True
        if coupledWithDef in individual.get_properties():
            individual.downstreamCoupling.append(True)
            if individual.name not in mapStatesCoupling:
                mapStatesCoupling[individual.name] = set()
            for entity in individual.coupledWith:
                mapStatesCoupling[individual.name].add(entity.name)
        else:
            individual.downstreamCoupling.append(False)
            individual.couplingHierarchy.append(0)
            mapStatesCoupling[individual.name] = set()
        listOfIndividualStates.append(individual)
        
        # Add state to the graph
        G.add_node(individual.name, type="state")
        node_labels[individual.name] = individual.name
        
        # Create composite states
        if useOldCompositeStates:
            create_composite_states(individual, visited, mapState2CompositeState)

reason_save_ontology(onto, fileSaveName)
# Add relationships between individuals as edges
edge_labels = {}
edge_colors = {}
all_individuals = list(onto.individuals())
# Function makes sure only downstreamCoupling = False states are connected
find_edges(all_individuals, G, actionsOfInterest)

# Sort the map
sorted_map = sort_map_by_set_length(mapStatesCoupling)
                        
mapStates2CompositeState = {}
mapStates2CombinedState = {}
mapOfAllCompositeStates = {}
allCompositeStatesBaseAction_ParentStates = {}

for key in sorted_map:
    print("Processing state: ", key)
    statesToFindCombOf = []
    composingStates = []
    allCompositeStates = []
    
    coupledWithStates = sorted_map[key]
    if len(coupledWithStates) == 0:
        mapStates2CompositeState[key] = []
        mapStates2CombinedState[key] = []
    else:
        for state in coupledWithStates:
            stateName = state
            stateAgent = get_corresponding_iri(onto, stateName).correspondingAgent[0]
            
            if mapStates2CompositeState[state] == []:
                # i.e. it makes no composite states
                composingStates.append([state, stateAgent.name])
            else:
                for st,agent,_ in mapStates2CompositeState[state]:
                    composingStates.append([st, stateAgent.name])

        print("Found Composing states for key: ", key)

        # respectiveAgents, noOfAgents = findNumberOfSameAgentStateCouplings(coupledWithStates)
        
        # Make all composite states
        for composingState, respectiveAgent in composingStates:
            bool_isGoodState = False
            composingStateOnto = get_corresponding_iri(onto, composingState)
            if composingStateOnto is None:
                print("Error")
                sys.exit()

            base = composingState
            if agentStateDef in composingStateOnto.is_a[0].ancestors():
                # composingState is a state
                if len(composingStateOnto.goodState) > 0:
                    bool_isGoodState = composingStateOnto.goodState[0]

            elif compositeStateDef in composingStateOnto.is_a[0].ancestors():
                # composingState is a composite state
                bool_isGoodState = composingStateOnto.goodState[0]
                base = allCompositeStatesBaseAction_ParentStates[composingState][1]

            else:
                print("composingState" , composingState, " is neither a state nor a composite state") 
                sys.exit()
            
            newCompositeStateName = "cSt_" + key + "_" + composingState
            newCompositeStateOnto = onto.compositeState(newCompositeStateName)
            newCompositeStateOnto.goodState.append(bool_isGoodState)
            allCompositeStates.append([newCompositeStateOnto.name, respectiveAgent, composingState])
            allCompositeStatesBaseAction_ParentStates[newCompositeStateName] = [base , key]
            # Add new composite state to the graph
            G.add_node(newCompositeStateOnto.name, type="compositeState")
            node_labels[newCompositeStateOnto.name] = newCompositeStateOnto.name

            newCompositeStateOnto.baseAgent.append(get_corresponding_iri(onto, respectiveAgent))
            newCompositeStateOnto.hasState.append(get_corresponding_iri(onto, key))
            # Add all the base states to the composite state
            if agentStateDef in composingStateOnto.is_a[0].ancestors():
                newCompositeStateOnto.hasState.append(composingStateOnto)
            elif compositeStateDef in composingStateOnto.is_a[0].ancestors():
                listOfhasState = composingStateOnto.hasState
                for state in listOfhasState:
                    newCompositeStateOnto.hasState.append(state)
                
        mapStates2CompositeState[key] = allCompositeStates   
        # Lets find all possible combinations of composite states
        defineRelationshipsBetweenCompositeStates(allCompositeStates, allCompositeStatesBaseAction_ParentStates, G)     
    print("Done with key: ", key)

# Save the ontology to a file
onto.save(file= fileSaveName, format="rdfxml")
reason_save_ontology(onto, fileSaveName)

# Add relationships between individuals as edges
addEdge = True
if addEdge:
    all_individuals = list(onto.individuals())
    find_edges(all_individuals, G, actionsOfInterest)

# Visualize the graph
vis = True
if vis:
    drawGraph(G)

# Example usage of the function
# start_node = "st_c22"
# end_node = "st_c34"  
# through_nodes = ["cSt_st_c23_cSt_st_B_2333C_st_lockC"]
# avoid_nodes = ["st_c21", "st_c12,st_c32", "st_c13", "st_c24", "st_c32", "st_c43"]  

# Example usage of the function
start_node = "st_c11" #"cSt_st_c11_st_B_1112C"  # Replace with actual node name
end_node = "st_c14" #"cSt_st_c13_st_B_1213O"    # Replace with actual node name
through_nodes = ["cSt_st_c12_st_rubbleNC"] #["cSt_st_c12_cSt_st_B_1213C_cSt_st_lock1O_st_lock2O"]
avoid_nodes = [] 

# find_shortest_path(G, start_node, end_node)

# Find all paths that pass through the specified nodes, avoid certain nodes, and have length constraints
paths = find_paths_through_nodes(G, start_node, end_node, through_nodes, avoid_nodes, min_length=3, max_length=8)

print(f"All paths from {start_node} to {end_node} passing through {through_nodes}, avoiding {avoid_nodes}, with length between 3 and 8")

currentStates = {}
currentStates["A_car"] = "st_c11"
currentStates["A_bulldozer"] =  "st_b11"

oriVehicle = "A_car"

# Function to find the indices of cs_callAnotherAgentAction in edge_labels
def find_indices_of_action(edge_labels, action_label):
    indices = [i for i, label in enumerate(edge_labels) if label == action_label]
    return indices

def get_repspective_vehicle(initial_state):
    if agentStateDef in initial_state.is_a[0].ancestors():
                V1 = initial_state.correspondingAgent[0]
    elif compositeStateDef in initial_state.is_a[0].ancestors():
        for state in initial_state.hasState:
            if motionAgentStateDef == state.is_a[0]:
                V1 = state.correspondingAgent[0]
    return V1
# Find pairs of cs_callAnotherAgentAction
action_label = "cs_callAnotherAgentAction"

def print_actionsForPath(path):
    # Get edge labels for the path
    edge_labels = []
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        edge_labels.append(edge_data.get('relationship', ''))
    print(f"Edge labels for the path: {edge_labels}") 
    return edge_labels


for path in paths:
    print("Path ====    ", path)
    edge_labels = print_actionsForPath(path)

    idx = find_indices_of_action(edge_labels, action_label)

    # for the action, if action is cs_callAnotherAgentAction then find the the two states and the vehicle performing the action
    # then find the path for the vehicle to come from original state to the final state
    for i in range(len(idx) - 1):
        p1 = idx[i]
        p2 = idx[i+1]
        init_st1 = get_corresponding_iri(onto,path[p1])
        V_I_st1 = get_repspective_vehicle(init_st1)
        final_st1 = get_corresponding_iri(onto,path[p1+1])
        V_F_st1 = get_repspective_vehicle(final_st1)

        init_st2 = get_corresponding_iri(onto,path[p2])
        V_I_st2 = get_repspective_vehicle(init_st2)
        final_st2 = get_corresponding_iri(onto,path[p2+1])
        V_F_st2 = get_repspective_vehicle(final_st2)
        
        if V_I_st1 == V_F_st2:
            # Find path for V2 to come from its initial state to final
            st2begin = currentStates[V_F_st1.name]
            st2end = final_st1.name
            newPath = find_paths_through_nodes(G, st2begin, st2end, [], [], min_length=0, max_length=8)
            shortestPath = newPath[0]

            print("Path for ", V_F_st1.name , " to come from ", st2begin, " to ", st2end, " is ", shortestPath)
            print_actionsForPath(shortestPath)

            currentStates[V_F_st1.name] = init_st2
            currentStates[V_F_st2.name] = final_st2

            print("Debug") 




























# from openai import OpenAI

# llama url https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicDhweXVhZ3d0ejJkZHR5eTAweXRjdWd4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNDE3MDgzM319fV19&Signature=hl0lat%7EfU745FmNVL402jy2BcO6xCGImWUdeGdWgxwtwakpp85idPBuMYiAje4%7EXZTXYhOCoXAQ8sih8F2NOCNN26wb0LwlmyQTuMw0HpRQ4MG8JppSbc6RzhnnFUppe%7E2L9LlkozaS76jLaSaAqqJQkePwNWUP5%7Es6FuHKaI1MArkGJZlhQNixOO4GGObzW2v1uhfpGy-rxKlmrtPyiZ-iwFm0CLo%7EYsoKJe748keJ1Ol9T6Hk8-lTmwjTRBm-hMBVrXlhL6jsgKPRv%7EHsDWkX7c2995R6JZ97q%7E01JRiPhUPiy8NaF4hbzUqGepj384CVX0cEIJMmRLdKZqO6Fbg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=497069582907745

# client = OpenAI(api_key='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')
# Set up your OpenAI API key
# openai.api_key = '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'

# def convert_swrl_to_python(swrl_rules):
#     prompt = f"Convert the following SWRL rules to a Python condition:\n\n{swrl_rules}\n\nPython condition:"

#     response = client.chat.completions.create(model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ])

#     python_condition = response.choices[0].message.content.strip()
#     return python_condition


def ask_ollama(prompt, model="llama3.1", system_prompt=None):
    """
    Send a prompt to Ollama and return the response.

    Args:
    prompt (str): The user's input prompt.
    model (str): The name of the Ollama model to use (default is "llama2").
    system_prompt (str, optional): A system prompt to set context (default is None).

    Returns:
    str: The AI's response.
    """
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except ollama._types.ResponseError as e:
        return f"Ollama error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"




# Read and convert SWRL rules to Python conditions
printRules = False
if printRules:
    print("SWRL Rules:")
    swrl_rules = ""
    for rule in onto.rules():
        print(rule)
        # result = ask_ollama(f"Convert the following SWRL rules to a Python condition:\n\n{rule}\n\nPython condition:")
        # print(result)
        swrl_rules += str(rule) + "\n"

# print("* Owlready2 * Warning: ignoring cyclic subclass of/subproperty of, involving:\n %s\n" % os, file = sys.stderr)

# Create an empty graph
# G_cyclicTest = nx.Graph()
# # Verify that cyclic dependencies are resolved
# cyclic_dependencies = []
# for cls in onto.classes():
#     G_cyclicTest.add_node(cls.name, type="class")
#     for super_cls in cls.is_a:
#         G_cyclicTest.add_node(super_cls.name, type="class")
#         G_cyclicTest.add_edge(cls.name, super_cls.name)

# # plot the graph
# nx.draw(G_cyclicTest, with_labels=True)
# plt.show()

# debug = False
# if debug:
#     for entity in onto.individuals():
#         if entity.storid == 409:
#             print(f"Entity with storid 409: {entity}")

# python_condition = convert_swrl_to_python(swrl_rules)
# print(f"Python condition: {python_condition}")



# Function to recursively find all possible downstream links
# def find_all_combinations(state, mapStatesCoupling, visited):
#     if state in visited:
#         return []
#     visited.add(state)
#     combinations = [[state]]
#     if state in mapStatesCoupling:
#         for linked_state in mapStatesCoupling[state]:
#             downstream_combinations = find_all_combinations(linked_state, mapStatesCoupling, visited)
#             for combo in downstream_combinations:
#                 combinations.append([state] + combo)
#     visited.remove(state)
#     return combinations


# # Function to generate all possible combinations for each key in the map
# def generate_combinations(mapStatesCoupling):
#     all_combinations = []
#     for state in mapStatesCoupling:
#         state_combinations = find_all_combinations(state, mapStatesCoupling, set())
#         all_combinations.extend(state_combinations)
#     return all_combinations




# mapStates2CompositeState = {}
# mapStates2CombinedState = {}
# mapOfAllCompositeStates = {}

# for key in sorted_map:
#     print("Processing state: ", key)
#     statesToFindCombOf = []
#     compState = []
#     coupledWithStates = sorted_map[key]
#     if len(coupledWithStates) == 0:
#         mapStates2CompositeState[key] = []
#         mapStates2CombinedState[key] = []
#     else:
#         for state in coupledWithStates:
#             if mapStates2CompositeState[state] == []:
#                 compState.append([state])
#             else:
#                 for st in mapStates2CompositeState[state]:
#                     if type(st) == list:
#                         compState.append([state] + st)
#                     else:
#                         compState.append([state, st])
            
#             if mapStates2CombinedState[state] == []:
#                 statesToFindCombOf.append(state)
#             else:
#                 for st in mapStates2CombinedState[state]:
#                     statesToFindCombOf.append(st)     
        
#         mapStates2CompositeState[key] = compState
#         mapStates2CombinedState[key] = statesToFindCombOf
#         print("Processed  state and printing for the key: ", key)

#         # Find all same state agents
#         respectiveAgents = {}
#         noOfAgents = 0
#         for state in coupledWithStates:
#             stringState = "*" + state
#             agent = onto.search_one(iri=stringState).[0]
#             if agent.name not in respectiveAgents:
#                 respectiveAgents[agent.name] = [state]
#                 noOfAgents += 1
#             else:
#                 respectiveAgents[agent.name].append(state)
        
#         allStatesAgent = {}
#         if noOfAgents > 1:
#             print("Booyahhh")
#             for agent in respectiveAgents.keys():
#                 statesOfEachAgent = []
#                 for state in respectiveAgents[agent]:
#                     # fl = False
#                     # for lock in mapStates2CombinedState[state]:
#                     for compState in mapStates2CompositeState[key]:
#                         if state == compState[0]:
#                             statesOfEachAgent.append(compState)
#                     #     fl = True
#                     # if not fl:
#                     #     statesOfEachAgent.append(state)
#                 allStatesAgent[agent] = statesOfEachAgent

#         else:
#             allStatesAgent[ next(iter(respectiveAgents))] = compState

#         allPossibleCombinations = []
#         allPossibleCombinationsName = []

#         if noOfAgents > 1:
#             for agent1 in allStatesAgent.keys():
#                 baseAgentName = "*"+next(iter(allStatesAgent))
#                 baseAgentOnto = onto.search_one(iri=baseAgentName)
#                 for comb1 in allStatesAgent[agent1]:
#                     if type(comb1) == list and len(comb1) > 0:
#                         comb1name = "_".join(comb1)
#                     else:
#                         comb1name = comb1
#                         print("error")
#                         sys.exit()
                    
#                     bool_isGoodState = False
#                     if len(comb1) > 1:
#                         # Then it is a composite state
#                         csStateName = "cSt_" + "_".join(comb1)
#                         csStateNameOnto = onto.search_one(iri="*" + csStateName)
#                         if csStateNameOnto is None:
#                             print("Error")
#                             sys.exit()
#                         bool_isGoodState = csStateNameOnto.goodState[0]
#                     else:
#                         # Then it is a state
#                         stateNameOnto = onto.search_one(iri="*" + comb1name)
#                         if len(stateNameOnto.goodState) > 0:
#                             bool_isGoodState = stateNameOnto.goodState[0]
                                    
#                     for agent2 in allStatesAgent.keys():
#                         if agent1 != agent2:
#                             for comb2 in allStatesAgent[agent2]:
#                                 if type(comb2) == list and len(comb2) > 0:
#                                     comb2name = "_".join(comb2)
#                                 else:
#                                     comb2name = comb2
#                                     print("error")
#                                     sys.exit()
                                
#                                 allPossibleCombinations.append([comb1,comb2])
#                                 compositeStateIndv = onto.compositeState(f"cSt_{key}_{comb1name}_{comb2name}")
#                                 allPossibleCombinationsName.append(f"cSt_{key}_{comb1name}_{comb2name}")
#                                 compositeStateIndv.baseAgent.append(baseAgentOnto)
#                                 compositeStateIndv..append(onto.search_one(iri="*"+key))
#                                 compositeStateIndv.goodState.append(bool_isGoodState)
#                                 mapOfAllCompositeStates[compositeStateIndv.name] = key 
                                
#                                 # Add state to the graph
#                                 G.add_node(compositeStateIndv.name, type="compositeState")
#                                 node_labels[compositeStateIndv.name] = compositeStateIndv.name

#         else:
#             for comb in allStatesAgent[next(iter(allStatesAgent))]:
#                 if type(comb) == list and len(comb) > 0:
#                     combname = "_".join(comb)
#                 else:
#                     combname = comb
#                     print("error")
#                     print("error")
#                     sys.exit()

#                 baseAgentName = "*" + next(iter(allStatesAgent))
#                 baseAgentOnto = onto.search_one(iri=baseAgentName)
#                 bool_isGoodState = False
                
#                 if len(comb) > 1:
#                     # Then it is a composite state
#                     csStateName = "cSt_" + "_".join(comb)
#                     csStateNameOnto = onto.search_one(iri="*" + csStateName)
#                     if csStateNameOnto is None:
#                         print("Error")
#                         sys.exit()
#                     combname = csStateName
#                     bool_isGoodState = csStateNameOnto.goodState[0]
#                 else:
#                     # Then it is a state
#                     stateNameOnto = onto.search_one(iri="*" + combname)
#                     if len(stateNameOnto.goodState) > 0:
#                         bool_isGoodState = stateNameOnto.goodState[0]
                
                
#                 allPossibleCombinations.append([combname])
#                 compositeStateIndv = onto.compositeState(f"cSt_{key}_{combname}")
#                 allPossibleCombinationsName.append(f"cSt_{key}_{combname}")
#                 compositeStateIndv.baseAgent.append(baseAgentOnto)
#                 compositeStateIndv.goodState.append(bool_isGoodState)
#                 compositeStateIndv..append(onto.search_one(iri="*"+key))
#                 mapOfAllCompositeStates[compositeStateIndv.name] = key
#                 # Add state to the graph
#                 G.add_node(compositeStateIndv.name, type="compositeState")
#                 node_labels[compositeStateIndv.name] = compositeStateIndv.name
                
#         allPossibleCombinations2 = list(itertools.product(*allStatesAgent.values()))
#         # Finding connections between newly defined composite states
#         for i in range(len(allPossibleCombinations)):
#             st1 = onto.search_one(iri="*"+"_".join(allPossibleCombinations[i]))
#             compSt1 = onto.search_one(iri="*"+allPossibleCombinationsName[i])
#             for j in range(len(allPossibleCombinations)):
#                 if i != j:
#                     st2 = onto.search_one(iri="*"+"_".join(allPossibleCombinations[j]))
#                     compSt2 = onto.search_one(iri="*"+allPossibleCombinationsName[j])
#                     csAction = False
                    
#                     if agentStateDef in st1.is_a[0].ancestors() and agentStateDef in st2.is_a[0].ancestors():
#                         # Both are states
#                         rel1 = st1
#                         rel2 = st2
#                         # rel1_likedNextState = onto.search_one(iri="*"+composingIndsList[i][1])
#                         # if compositeStateDef == rel1_likedNextState.is_a[0] and not rel1_likedNextState.goodState[0]:
#                         #     continue
#                         # rel2_likedNextState = onto.search_one(iri="*"+composingIndsList[j][1])
#                         # if compositeStateDef == rel2_likedNextState.is_a[0] and not rel2_likedNextState.goodState[0]:
#                         #     continue
                    
                    
#                     elif agentStateDef in st1.is_a[0].ancestors() and compositeStateDef in st2.is_a[0].ancestors():
#                         # st1 is state and st2 is composite state
#                         rel1 = st1
#                         rel2 = onto.search_one(iri="*" + mapOfAllCompositeStates[st2.name])
                        
#                         if st2.goodState[0] == False or st1.goodState[0] == False:
#                             continue

#                     elif compositeStateDef in st1.is_a[0].ancestors() and agentStateDef in st2.is_a[0].ancestors():
#                         # st2 is state and st1 is composite state
#                         rel1 = onto.search_one(iri="*" + mapOfAllCompositeStates[st1.name])
#                         rel2 = st2 
#                         if st1.goodState[0] == False or st2.goodState[0] == False:
#                             continue
#                     else:
#                         # Both are composite states
#                         rel1 = onto.search_one(iri="*"+  mapOfAllCompositeStates[st1.name])
#                         rel2 = onto.search_one(iri="*"+  mapOfAllCompositeStates[st2.name])
#                         if rel1.name == rel2.name:
#                             continue    
#                         if st1.goodState[0] == False or st2.goodState[0] == False:
#                             continue
                        
#                         csAction = True
                    
#                     # Now check if two states are connected by a object property as mentioned in actions
#                     for action in actions:
#                         actionAdd = "cs_" + action + "Please"
#                         if csAction:
#                             action = "cs_" + action + "Please"
#                         if rel2 in getattr(rel1,action):
#                             print(compSt1.name, actionAdd, compSt2.name)
#                             if compSt2 not in getattr(compSt1, actionAdd):
#                                 getattr(compSt1, actionAdd).append(compSt2)
#                                 print("done")

#                                 G.add_edge(compSt1.name, compSt2.name, relationship=action)
#                                 edge_labels[(compSt1.name, compSt2.name)] = action


                    
                        
                    
                    
                    


            
#             # allposs = allPossibleCombinations[i]
#             # compStName = allPossibleCombinationsName[i]
#             # for st in allposs:
#             #     agent,value = find_agent_value(state)
#             #     #Find name of the composite state
#             #     compStateOnto = onto.search_one(iri=("*" + compStName))
#             #     if compStateOnto is None:
#             #         print("Error")

#             #     compStateOnto.corrospondsTo.append(value)
#             #     print("something something")
#         print("done with all possibl combinations")
    

# # Ensure all individuals in the ontology are different
# all_individuals = list(onto.individuals())
# AllDifferent(all_individuals)

# with onto:
#     sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True)
#     print("Reasoning completed")

# # Save the ontology to a file
# onto.save(file="/Users/bhamri/Desktop/Ontology/owlFiles/saved_ontologyV2.owl", format="rdfxml")
