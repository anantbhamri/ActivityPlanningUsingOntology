import os
from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt
import ollama
import sys
import time
import plotly.graph_objects as go
import itertools
import pickle
import joblib
import globals

from GraphGenerationPlanning import get_corresponding_iri
from GraphGenerationPlanning import define_compositeStates
from GraphGenerationPlanning import add_nomianlActionEdgesforAgentIndividual
from GraphGenerationPlanning import identifyCorrespondingAgentAndValue
from GraphGenerationPlanning import find_all_paths_with_conflict_check_DFS
from GraphGenerationPlanning import find_all_paths_with_conflict_check_Dijkstras
from GraphGenerationPlanning import checkIfOutgoingEdgesHaveTimePassageAction
# from GraphGenerationPlanning import find_paths_through_nodes
from GraphGenerationPlanning import find_all_paths_with_conflict_check_DijkstrasV1

# Initialize the global variable
globals.start_timeInitial = time.time()

start_timeInitial = globals.start_timeInitial

# Plot Graph function
def plotGraphCustomMethod(G):
    pos = nx.spring_layout(G)  # positions for all nodes

    # Create edge traces
    edge_trace = go.Scatter(
        x=(),
        y=(),
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create node traces
    node_trace = go.Scatter(
        x=(),
        y=(),
        text=(),
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (str(node),)

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Network Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Show the interactive plot
    fig.show()


visualizeGraphNominalGraph = False
visualizeGraphCompleteGraph = False

usePickleFile = False
# Name of the ontology file with the path and where to save it
# 1 for ruble eaxmple, 2 for nuclear plant example, 3 for nuclear plant example with multiple vehicles
testCase = 1
if testCase == 1:
    onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach2_NuclearPlantmultiVehicleV3_simple.owl").load()
    # Define the IC
    IC = {}
    IC["A_Veh1VO"] = 'st_Veh1Road5'
    IC["A_Veh2B"] = 'st_Veh2Road5'
    IC["A_Valve"] = 'st_VC'
    IC["A_Rubble_R5"] = 'st_RubbleC_R5'
    IC["A_Rubble_R6"] = 'st_RubbleNC_R6'
    IC["A_Rubble_R7"] = 'st_RubbleC_R7'

    FS = {}
    # FS["A_Valve"] = 'st_VO'
    # agentWhichNeedsToBeControlled = "A_Valve"
    FS["A_Veh1VO"] = 'st_Veh1Road7'
    agentWhichNeedsToBeControlled = "A_Veh1VO"

if testCase == 2:
    onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_NuclearPlantmultiVehicle.owl").load()
    # Define the IC
    IC = {}
    IC["A_CoolingPump"] = 'st_CPOff'
    IC["A_FuelAmount"] = 'st_FE'
    IC["A_FuelSensor"] = 'st_FND'
    IC["A_FuelTruckConnection"] = 'st_FTNc'
    IC["A_Generator"] = 'st_GOff'
    IC["A_LinePower"] = 'st_LPOff'
    IC["A_Reactor"] = 'st_RH'
    IC["A_Valve"] = 'st_VC'

    FS = {}
    FS["A_Reactor"] = 'st_RL'
    agentWhichNeedsToBeControlled = "A_Reactor"

if testCase == 3:
    onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_NuclearPlantmultiVehicleV3_simple.owl").load()
    # Define the IC
    IC = {}
    IC["A_CoolingPump"] = 'st_CPOff'
    IC["A_FuelAmount"] = 'st_FE'
    IC["A_FuelSensor"] = 'st_FND'
    IC["A_FuelTruckConnection"] = 'st_FTNc'
    IC["A_Generator"] = 'st_GOff'
    IC["A_LinePower"] = 'st_LPOff'
    IC["A_Reactor"] = 'st_RH'
    IC["A_Rubble_Int1"] = 'st_RubbleNC_Int1'
    IC["A_Rubble_Int2"] = 'st_RubbleNC_Int2'
    IC["A_Rubble_Int3"] = 'st_RubbleNC_Int3'
    IC["A_Rubble_Int4"] = 'st_RubbleNC_Int4'
    IC["A_Rubble_Int5"] = 'st_RubbleNC_Int5'
    IC["A_Rubble_Int6"] = 'st_RubbleNC_Int6'
    IC["A_Rubble_Int7"] = 'st_RubbleNC_Int7'
    IC["A_Rubble_Int8"] = 'st_RubbleNC_Int8'
    IC["A_Rubble_Int9"] = 'st_RubbleNC_Int9'
    IC["A_Rubble_R1"] = 'st_RubbleNC_R1'
    IC["A_Rubble_R2"] = 'st_RubbleNC_R2'
    IC["A_Rubble_R3"] = 'st_RubbleNC_R3'
    IC["A_Rubble_R4"] = 'st_RubbleNC_R4'
    IC["A_Rubble_R5"] = 'st_RubbleNC_R5'
    IC["A_Rubble_R6"] = 'st_RubbleNC_R6'
    IC["A_Rubble_R7"] = 'st_RubbleNC_R7'
    IC["A_Rubble_R8"] = 'st_RubbleNC_R8'
    IC["A_Rubble_R9"] = 'st_RubbleNC_R9'
    IC["A_Rubble_R10"] = 'st_RubbleNC_R10'
    IC["A_Rubble_R11"] = 'st_RubbleNC_R11'
    IC["A_Rubble_R12"] = 'st_RubbleNC_R12'
    IC["A_Rubble_R13"] = 'st_RubbleNC_R13'
    IC["A_Rubble_R14"] = 'st_RubbleNC_R14'
    IC["A_Rubble_R15"] = 'st_RubbleNC_R15'
    IC["A_Rubble_R16"] = 'st_RubbleNC_R16'
    IC["A_Valve"] = 'st_VC'
    IC["A_Veh1VO"] = 'st_Veh1Inter5'
    IC["A_Veh2B"] = 'st_Veh2Road10'
    IC["A_Veh3FT"] = 'st_Veh3Road11'
    IC["A_Veh4PT"] = 'st_Veh4Road11'
    IC["A_Veh5Ins"] = 'st_Veh5Road11'



    FS = {}
    FS["A_Reactor"] = 'st_RL'
    agentWhichNeedsToBeControlled = "A_Reactor"
    # FS["A_Veh1VO"] = 'st_Veh1Road7'
    # agentWhichNeedsToBeControlled = "A_Veh1VO"

if testCase == 4:
    onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach2_NuclearPlantmultiVehicleV3_verysimple.owl").load()
    # Define the IC
    IC = {}
    IC["A_CoolingPump"] = 'st_CPOff'
    IC["A_FuelAmount"] = 'st_FE'
    IC["A_FuelSensor"] = 'st_FND'
    IC["A_FuelTruckConnection"] = 'st_FTNc'
    IC["A_Generator"] = 'st_GOff'
    IC["A_LinePower"] = 'st_LPOff'
    IC["A_Reactor"] = 'st_RH'
    IC["A_Rubble_Int8"] = 'st_RubbleNC_Int8'
    IC["A_Rubble_R11"] = 'st_RubbleNC_R11'
    IC["A_Rubble_R12"] = 'st_RubbleNC_R12'
    IC["A_Valve"] = 'st_VC'
    IC["A_Veh1VO"] = 'st_Veh1Road11'
    IC["A_Veh2B"] = 'st_Veh2Road11'
    
    FS = {}
    FS["A_Reactor"] = 'st_RL'
    agentWhichNeedsToBeControlled = "A_Reactor"

    # FS["A_Veh1VO"] = 'st_Veh1Road12'
    # agentWhichNeedsToBeControlled = "A_Veh1VO"




# onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1.owl").load()
# onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/reach_agentStateSpace_approach1_multiVehicle.owl").load()

fileSaveName = "/Users/bhamri/Desktop/Ontology/owlFiles/saved_ontologyV2.owl"

# Set the Java options to increase the heap space
os.environ['JAVA_OPTS'] = '-Xmx16G'

# Definations of the ontology iri's that might be of interest
baseIRIOnto = onto.base_iri
agentStateDef = get_corresponding_iri(onto, "agentState")
agentClassDef = get_corresponding_iri(onto, "agentClass")
compositeStateDef = get_corresponding_iri(onto, "compositeState")
coupledWithDef = get_corresponding_iri(onto, "coupledWith")
downstreamCouplingDef = get_corresponding_iri(onto, "downstreamCoupling")
sameStateAgentDef = get_corresponding_iri(onto, "sameStateAgent")
agentClass = get_corresponding_iri(onto, "agentClass")
motionAgentStateDef = get_corresponding_iri(onto, "motionAgentState")
clearRubbleActionPleaseDef = get_corresponding_iri(onto, "cs_clearRubbleActionPlease")

map_agent2nominalActions = {}
map_agent2nominalActions["controlPump"] = ["timePassageAction",]
map_agent2nominalActions["fuelAmount"] = ["timePassageAction",]
map_agent2nominalActions["fuelLevelSensor"] = ["timePassageAction",]
map_agent2nominalActions["fuelTruckConnection"] = ["disconnectFuelTruckAction","connectFuelTruckAction",]
map_agent2nominalActions["generator"] = ["timePassageAction", "turnOnGeneratorAction",]
map_agent2nominalActions["linePower"] = ["powerCutEventReach",]
map_agent2nominalActions["reactorCore"] = ["timePassageAction",]
map_agent2nominalActions["valve"] = ["closeVentAction", "openVentAction",]
map_agent2nominalActions["motionAgent"] = ["goWAction", "goEAction", "goSAction", "goNAction",]
map_agent2nominalActions["rubble"] = ["clearRubbleAction",]

reason = True
if not usePickleFile:
# Reasoning on the ontology to get the inferred properties mainly the nomianl actions
    if reason:
        # Measure the time taken for reasoning
        start_time = time.time()
        # Perform reasoning
        
        # Increase Java heap space and run the reasoner
        with onto:
            try:
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
                print("Reasoning completed")
            except Exception as e:
                print(f"Reasoning failed: {e}")
                sys.exit()
        
        end_time = time.time()
        reasoning_time = end_time - start_time
        print(f"Reasoning took {reasoning_time:.2f} seconds")

    # Now we want to get all the individuals that are agents(agentClass) from the ontology . 
    # Then for each agent we want to look at all the possible states(agentState), 
    # look at the base nominal action that can be performed in each state and the constraining states.  
    # Then create composite states which are a combination of the agentState and the constraining states and the not of the constraining states(bar).

    # Define all the agent and values for the individuals
    onto = identifyCorrespondingAgentAndValue(onto, agentStateDef)

    # Find all the agent class
    individualsOfAgentClass = list(agentClassDef.instances())

    noOfCompositeStates = 0
    map_compositeState2hasStates = {}
    map_state2compositeState = {}
    map_agent2CompositeStates = {}
    # Create a graph for the composite states
    G = nx.DiGraph()
    node_labels = {}
    edge_labels = {}
    edge_actingAgent = {}
    edge_changingStates = {}

    for agent in individualsOfAgentClass:
        
        noNominalEdgesAdded = G.number_of_edges()
        noNominalNodesAdded = G.number_of_nodes()

        # Find the states of an individual
        # This is list of individuls that have the property corrospondsTo 
        allStates = getattr(agent,"corrospondedBy")
        
        map_agent2CompositeStates[agent.name] = []
        # Create the composite states for the agent 
        # These are the nodes in the graph
        for state in allStates:
            # Find the constraining states of the state
            noOfCompositeStates, map_compositeState2hasStates, map_state2compositeState, G, node_labels, allNewStates = define_compositeStates(state, noOfCompositeStates, map_compositeState2hasStates, map_state2compositeState, G, node_labels )
            map_agent2CompositeStates[agent.name].extend(allNewStates)
        # Find the type of the agentClass
        agentType = agent.is_a[0]
        # Find the nominal action of the agent
        possibleNominalActions = map_agent2nominalActions[agentType.name]
        
        # Add edges (nominal actions) between the nodes
        G, edge_labels, edge_actingAgent = add_nomianlActionEdgesforAgentIndividual(allStates, map_state2compositeState, map_compositeState2hasStates, possibleNominalActions, G, edge_labels, edge_actingAgent, agent.name, onto)
        noNominalEdgesAdded = G.number_of_edges() - noNominalEdgesAdded
        noNominalNodesAdded = G.number_of_nodes() - noNominalNodesAdded
        print("Done creating the composite states for the agent: ", agent.name, " and added ", noNominalNodesAdded , " nodes and ", noNominalEdgesAdded, " edges for the nominal actions between the composite states")

    end_timeNominalGraph = time.time()
    print(f"Creating the nominal graph took {end_timeNominalGraph - start_timeInitial:.2f} seconds")

    # print the size of the graph+
    print("The number of nodes in the graph is: ", G.number_of_nodes())
    print("The number of edges in the graph is: ", G.number_of_edges())

    if visualizeGraphNominalGraph:
        # visualize graph
        plotGraphCustomMethod(G)
        nx.draw(G, with_labels=True)
        plt.show()

    createACompleteGraph = True
    if createACompleteGraph:
        # Add the nodes between not composite state of different agents
        for key1 in (map_agent2CompositeStates):
            allCompositeStates1 = (map_agent2CompositeStates[key1])
            
            for key2 in (map_agent2CompositeStates):
                if key1 == key2:
                    continue
                allCompositeStates2 = (map_agent2CompositeStates[key2])
                # find pairs of all compostie states
                for compositeState1 in allCompositeStates1:
                    for compositeState2 in allCompositeStates2:
                        # Add an edge between the composite states only if 
                        hasStates1 = map_compositeState2hasStates[compositeState1]
                        hasStates2 = map_compositeState2hasStates[compositeState2]
                        
                        agentsCS1 = G.nodes[compositeState1]['agents']
                        agentsCS2 = G.nodes[compositeState2]['agents']

                        hasAtleastOneCommonAgent = False
                        allCommonAgentsHaveSameState = False
                        # Find a list of all same agents and their corresponding states
                        sameAgentStates = []
                        for agent1 in agentsCS1:
                            if agent1 in agentsCS2:
                                hasAtleastOneCommonAgent = True
                                # Find index of the agent1 in both agentsCS1 and agentsCS2
                                # Find the corresponding states
                                index1 = agentsCS1.index(agent1)
                                index2 = agentsCS2.index(agent1)
                                state1 = hasStates1[index1]
                                state2 = hasStates2[index2]
                                if state1 == state2:
                                    allCommonAgentsHaveSameState = True
                                
                                else:
                                    if get_corresponding_iri(onto,agent1).is_a[0] == motionAgentStateDef:
                                        print("Agent is a motion agent")
                                    # check if state1 has string "_bar" in it
                                    if "_bar" in state1 and "_bar" not in state2:
                                        setOfStatesOfAgent = set(get_corresponding_iri(onto,agent1).corrospondedBy)
                                        barOfState1 = state1[:-4]
                                        # remove the barOfState2 from the set
                                        bar_setOfStatesOfAgent = setOfStatesOfAgent - {get_corresponding_iri(onto,barOfState1)}
                                        # check if state1 is in the set
                                        if get_corresponding_iri(onto,state2) in bar_setOfStatesOfAgent:
                                            allCommonAgentsHaveSameState = True
                                        else:
                                            allCommonAgentsHaveSameState = False
                                            break

                                    elif "_bar" in state2 and "_bar" not in state1:
                                        setOfStatesOfAgent = set(get_corresponding_iri(onto,agent1).corrospondedBy)
                                        barOfState2 = state2[:-4]
                                        # remove the barOfState2 from the set
                                        bar_setOfStatesOfAgent = setOfStatesOfAgent - {get_corresponding_iri(onto,barOfState2)}
                                        # check if state1 is in the set
                                        if get_corresponding_iri(onto,state1) in bar_setOfStatesOfAgent:
                                            allCommonAgentsHaveSameState = True
                                        else:
                                            allCommonAgentsHaveSameState = False
                                            break
                                    else:
                                        allCommonAgentsHaveSameState = False
                                        break

                        if hasAtleastOneCommonAgent and allCommonAgentsHaveSameState:
                            cs1HasTimePassageAction = checkIfOutgoingEdgesHaveTimePassageAction(G, compositeState1)
                            if cs1HasTimePassageAction:
                                continue
                            G.add_edge(compositeState1, compositeState2, action = 'callDifferentAgent', cost = 100, actingAgent = None)
                            edge_labels[(compositeState1, compositeState2)] = "noAction"
                            edge_actingAgent[(compositeState1, compositeState2)] = "callingOtherAgent"
                            edge_changingStates[(compositeState1, compositeState2)] = "noAction"

    end_timeCompleteGraph = time.time()
    print(f"Creating the complete graph took {end_timeCompleteGraph - end_timeNominalGraph:.2f} seconds")
    
    # print the size of the graph+
    print("The number of nodes in the graph is: ", G.number_of_nodes())
    print("The number of edges in the graph is: ", G.number_of_edges())

    if visualizeGraphCompleteGraph:
        # visualize complete graph
        plotGraphCustomMethod(G)
        nx.draw(G, with_labels=True)
        plt.show()

    # Save each variable individually to a pickle file
    with open('graph_data.pkl', 'wb') as f:
        pickle.dump(G, f)
        pickle.dump(noOfCompositeStates, f)
        pickle.dump(map_compositeState2hasStates, f)
        pickle.dump(map_state2compositeState, f)
        pickle.dump(map_agent2CompositeStates, f)
        pickle.dump(node_labels, f)
        pickle.dump(edge_labels, f)
        pickle.dump(edge_actingAgent, f)
        pickle.dump(edge_changingStates, f)
    print("Data saved successfully to 'graph_data.pkl'")

else:
    if reason:
        # Measure the time taken for reasoning
        start_time = time.time()
        # Perform reasoning
        
        # Increase Java heap space and run the reasoner
        with onto:
            try:
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
                print("Reasoning completed")
            except Exception as e:
                print(f"Reasoning failed: {e}")
                sys.exit()
        
        end_time = time.time()
        reasoning_time = end_time - start_time
        print(f"Reasoning took {reasoning_time:.2f} seconds")

    # Define all the agent and values for the individuals
    onto = identifyCorrespondingAgentAndValue(onto, agentStateDef)

    # Load the data from the pickle file
    with open('graph_data.pkl', 'rb') as f:
        G = pickle.load(f)
        noOfCompositeStates = pickle.load(f)
        map_compositeState2hasStates = pickle.load(f)
        map_state2compositeState = pickle.load(f)
        map_agent2CompositeStates = pickle.load(f)
        node_labels = pickle.load(f)
        edge_labels = pickle.load(f)
        edge_actingAgent = pickle.load(f)
        edge_changingStates = pickle.load(f)

        print("Data loaded successfully from 'graph_data.pkl'")
        print("The number of nodes in the graph is: ", G.number_of_nodes())
        print("The number of edges in the graph is: ", G.number_of_edges())


# sys.exit()

def find_all_paths_dfs(G, start, end, path=[], all_paths=[]):
    path = path + [start]
    if start == end:
        all_paths.append(path)
        return all_paths
    if start not in G:
        return all_paths
    for node in G.neighbors(start):
        if node not in path:
            find_all_paths_dfs(G, node, end, path, all_paths)
    return all_paths

# Initialize global variables if needed
longest_path = []

def process_path(path):
    # Process the path here
    print(f"Found path: {path}")
    
    # You can perform any operations you need on the path
    path_length = len(path) - 1
    print(f"Path length: {path_length}")
    
    # You could also store important information about the path
    # For example, you might keep track of the longest path
    global longest_path
    if path_length > len(longest_path):
        longest_path = path



def find_shortest_path_dijkstra(G, start, end, weight='cost'):
    try:
        for path in nx.all_simple_paths(G, source=start, target=end):
            process_path(path)
        shortest_path = nx.shortest_path(G, source=start, target=end, weight=weight)
        shortest_path_cost = nx.shortest_path_length(G, source=start, target=end, weight=weight)
        return shortest_path, shortest_path_cost
    except nx.NetworkXNoPath:
        return None, float('inf')
        

def find_all_simple_paths(G, start, end):
    try:
        all_paths = list(nx.all_simple_paths(G, source=start, target=end))
        return all_paths
    except nx.NetworkXNoPath:
        return []

# Plan on the graph
createReachabilityFromICGraph = True
if createReachabilityFromICGraph == True:

    listOfStartingNodes = map_state2compositeState[IC[agentWhichNeedsToBeControlled]]
    startingNode = None
    for node in listOfStartingNodes:
        allCorrespondingStatesInIC = False
        listOfStates = map_compositeState2hasStates[node]
        listOfAgents = G.nodes[node]['agents']
        for idx in range(1,len(listOfStates)):
            state = listOfStates[idx]
            agentOfState = listOfAgents[idx]
            # check if the state is a bar state
            if "_bar" in state:
                setOfStatesOfAgent = set(get_corresponding_iri(onto,agentOfState).corrospondedBy)
                barOfState1 = state[:-4]
                # remove the barOfState2 from the set
                bar_setOfStatesOfAgent = setOfStatesOfAgent - {get_corresponding_iri(onto,barOfState1)}
                # check if state1 is in the set
                if get_corresponding_iri(onto,IC[agentOfState]) in bar_setOfStatesOfAgent: 
                    allCorrespondingStatesInIC = True
                else:
                    allCorrespondingStatesInIC = False
                    break

            else:
                if IC[agentOfState] == get_corresponding_iri(onto,state).name:
                    allCorrespondingStatesInIC = True
                else:
                    allCorrespondingStatesInIC = False
                    break
        
        if allCorrespondingStatesInIC:
            startingNode = node    
            break

    if startingNode == None:
        print("Starting node not found")
        sys.exit()

    listOfEndingNodes = map_state2compositeState[FS[agentWhichNeedsToBeControlled]]
    allPossiblePaths = []
    for endingNode in listOfEndingNodes:
        useStdFunction = False
        if useStdFunction:
            # Find the shortest path using Dijkstra's algorithm
            # shortest_path, shortest_path_cost = find_shortest_path_dijkstra(G, startingNode, endingNode)
            # print(f"Shortest path: {shortest_path}, Cost: {shortest_path_cost}")

            # Find all simple paths
            all_paths = find_all_simple_paths(G, startingNode, endingNode)
            print("All simple paths:")
            for path in all_paths:
                print(path)
            # Find all paths using DFS
            # all_paths = find_all_paths_dfs(G, startingNode, endingNode)
            # print("All paths:")
            # for path in all_paths:
            #     print(path)

        allPathsToNode = find_all_paths_with_conflict_check_DFS(G, startingNode, endingNode, map_compositeState2hasStates, IC,onto,FS, agentWhichNeedsToBeControlled)
        # allPathsToNode = find_all_paths_with_conflict_check_Dijkstras(G, startingNode, endingNode, map_compositeState2hasStates, IC,onto, FS,agentWhichNeedsToBeControlled)
        # allPathsToNode = find_all_paths_with_conflict_check_DijkstrasV1(G, startingNode, endingNode, map_compositeState2hasStates, IC,onto)
        allPossiblePaths.extend(allPathsToNode)
    
    # sort the paths based on the cost and print the minimum cost path
    allPossiblePaths.sort(key=lambda x: x[1])
    print("Minimum cost actions to reach the final state")
    for path, cost, actions in allPossiblePaths:
        print()
        print()
        print(allPossiblePaths[0][2])
        for eachAction in actions:
            # if "call Agent " in eachAction:
                # continue
            print(f"Action: {eachAction}")
        print(f"Path: {path}, Cost: {cost}")
        print()
        print()

timeTaken = time.time() - start_timeInitial
print(f"Total time taken: {timeTaken:.2f} seconds")    

sys.exit()

start_node = "st_c11" #"cSt_st_c11_st_B_1112C"  # Replace with actual node name
end_node = "st_c14" #"cSt_st_c13_st_B_1213O"    # Replace with actual node name
through_nodes = ["cSt_st_c12_st_rubbleNC"] #["cSt_st_c12_cSt_st_B_1213C_cSt_st_lock1O_st_lock2O"]
avoid_nodes = [] 

# find_shortest_path(G, start_node, end_node)

# Find all paths that pass through the specified nodes, avoid certain nodes, and have length constraints
# paths = find_paths_through_nodes(G, start_node, end_node, through_nodes, avoid_nodes, min_length=3, max_length=8)


