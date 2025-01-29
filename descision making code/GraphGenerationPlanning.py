# Functions for ontology reasoning and graph planning

import os
from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt
import ollama
import sys
import time
import plotly.graph_objects as go
import itertools
import heapq
from copy import deepcopy
import globals

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

def find_agent_value(state, onto):
    agentClass = get_corresponding_iri(onto, "agentClass")
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


def identifyCorrespondingAgentAndValue(onto, agentStateDef):
    # Add individual states as nodes and define their hierarchy
    for individual in onto.individuals():
        # Add the individuals which are states as nodes
        indvClass = individual.is_a[0]
        if agentStateDef in indvClass.ancestors():
            # find_agent_value
            agentName, agentValue = find_agent_value(individual.name, onto)
            individual.correspondingAgent.append(agentName)
            individual.correspondingValue.append(agentValue)
    return onto


def define_compositeStates(state, noOfCompositeStates, map_compositeState2hasStates, map_state2compositeState, G, node_labels):
    allNewStates = []
    listOfAgents =  []
    listOfAgents.append(state.correspondingAgent[0].name)
    if state.name not in map_state2compositeState:
        map_state2compositeState[state.name] = []

    useDemorgansLaw = True
    makeExtra = True

    constrainingStatesOr = getattr(state, "constrainingStatesOr")
    constrainingStatesAnd = getattr(state, "constrainingStatesAnd")
    
    if len(constrainingStatesOr) > 2 or len(constrainingStatesAnd) > 2:
        print("More than 2 constraining states test")

    if len(constrainingStatesOr) == 0 and len(constrainingStatesAnd) == 0:
        noOfCompositeStates += 1
        compositeStateName = "cs_" + str(noOfCompositeStates)
        map_compositeState2hasStates[compositeStateName] = [state.name]
        map_state2compositeState[state.name].append(compositeStateName)
        allNewStates.append(compositeStateName)
        G.add_node(compositeStateName, type="state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name, agentTypeBar = False)
        node_labels[compositeStateName] = compositeStateName

    if len(constrainingStatesOr) == 1 or len(constrainingStatesAnd) == 1:
        noOfCompositeStates += 1
        compositeStateName = "cs_" + str(noOfCompositeStates)
        allNewStates.append(compositeStateName)
        map_state2compositeState[state.name].append(compositeStateName)
        map_compositeState2hasStates[compositeStateName] = [state.name]
        if len(constrainingStatesOr) == 1:
            constrainingState = constrainingStatesOr[0]
        if len(constrainingStatesAnd) == 1:
            constrainingState = constrainingStatesAnd[0]

        map_compositeState2hasStates[compositeStateName].append(constrainingState.name)
        listOfAgents.append(constrainingState.correspondingAgent[0].name)
        G.add_node(compositeStateName, type="state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name, agentTypeBar = False)
        node_labels[compositeStateName] = compositeStateName

        noOfCompositeStates += 1
        compositeStateName = "cs_" + str(noOfCompositeStates)
        allNewStates.append(compositeStateName)
        map_state2compositeState[state.name].append(compositeStateName)
        map_compositeState2hasStates[compositeStateName] = [state.name]       
        map_compositeState2hasStates[compositeStateName].append(constrainingState.name + "_bar")
        G.add_node(compositeStateName, type="state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name, agentTypeBar = True)
        node_labels[compositeStateName] = compositeStateName


    elif len(constrainingStatesOr) > 1 and len(constrainingStatesAnd) == 0:
        if useDemorgansLaw:     
            # We will have a pair of composite states for each constraining state
            noOfCompositeStates += 1
            compositeStateName = "cs_" + str(noOfCompositeStates)
            map_compositeState2hasStates[compositeStateName] = [state.name]
            map_state2compositeState[state.name].append(compositeStateName)
            allNewStates.append(compositeStateName)
            for constrainingState in constrainingStatesOr:
                map_compositeState2hasStates[compositeStateName].append(constrainingState.name)
                listOfAgents.append(constrainingState.correspondingAgent[0].name)
            G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = False)
            node_labels[compositeStateName] = compositeStateName
            

            noOfCompositeStates += 1
            compositeStateName = "cs_" + str(noOfCompositeStates)
            map_compositeState2hasStates[compositeStateName] = [state.name]
            map_state2compositeState[state.name].append(compositeStateName)
            allNewStates.append(compositeStateName)
            for constrainingState in constrainingStatesOr:
                map_compositeState2hasStates[compositeStateName].append(constrainingState.name + "_bar")
            G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name, agentTypeBar = True)
            node_labels[compositeStateName] = compositeStateName

            if makeExtra:
                for i in range(len(constrainingStatesOr)):
                    noOfCompositeStates += 1
                    compositeStateName = "cs_" + str(noOfCompositeStates)
                    map_compositeState2hasStates[compositeStateName] = [state.name] 
                    map_state2compositeState[state.name].append(compositeStateName)
                    allNewStates.append(compositeStateName)
                    for j in range(len(constrainingStatesOr)):
                        if i == j:
                            map_compositeState2hasStates[compositeStateName].append(constrainingStatesOr[j].name + "_bar")
                        else:
                            map_compositeState2hasStates[compositeStateName].append(constrainingStatesOr[j].name)
                    G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = False)
                    node_labels[compositeStateName] = compositeStateName



        else:
            # We will have a pair of composite states for each constraining state
            for idx_constrainingState in range(0,len(constrainingStatesOr)):
                listOfAgents =  []
                listOfAgents.append(state.correspondingAgent[0].name)

                constrainingState = constrainingStatesOr[idx_constrainingState]
                listOfAgents.append(constrainingState.correspondingAgent[0].name)

                noOfCompositeStates += 1
                compositeStateName = "cs_" + str(noOfCompositeStates)
                allNewStates.append(compositeStateName)
                map_state2compositeState[state.name].append(compositeStateName)
                
                
                map_compositeState2hasStates[compositeStateName] = [state.name]
                map_compositeState2hasStates[compositeStateName].append(constrainingState.name )
                
                G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = False)
                node_labels[compositeStateName] = compositeStateName
            
                # Creating the bar state
                noOfCompositeStates += 1
                compositeStateName = "cs_" + str(noOfCompositeStates)
                allNewStates.append(compositeStateName)
                map_state2compositeState[state.name].append(compositeStateName)
                
                map_compositeState2hasStates[compositeStateName] = [state.name]
                map_compositeState2hasStates[compositeStateName].append(constrainingState.name + "_bar")
                
                G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = True)
                node_labels[compositeStateName] = compositeStateName
            
    elif len(constrainingStatesOr) == 0 and len(constrainingStatesAnd) > 1:
        # Use Demorgans law to find the combinations
        noOfCompositeStates += 1
        compositeStateName = "cs_" + str(noOfCompositeStates)
        map_compositeState2hasStates[compositeStateName] = [state.name]
        map_state2compositeState[state.name].append(compositeStateName)
        allNewStates.append(compositeStateName)
        for x in constrainingStatesAnd:
            map_compositeState2hasStates[compositeStateName].append(x.name)
            listOfAgents.append(x.correspondingAgent[0].name)
        G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = False)
        node_labels[compositeStateName] = compositeStateName
        
        if useDemorgansLaw:
            for i in range(len(constrainingStatesAnd)):
                noOfCompositeStates += 1
                compositeStateName = "cs_" + str(noOfCompositeStates)
                map_compositeState2hasStates[compositeStateName] = [state.name] 
                map_state2compositeState[state.name].append(compositeStateName)
                allNewStates.append(compositeStateName)
                for j in range(len(constrainingStatesAnd)):
                    if i == j:
                        map_compositeState2hasStates[compositeStateName].append(constrainingStatesAnd[j].name + "_bar")
                    else:
                        map_compositeState2hasStates[compositeStateName].append(constrainingStatesAnd[j].name)
                G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = True)
                node_labels[compositeStateName] = compositeStateName

            if makeExtra:
                noOfCompositeStates += 1
                compositeStateName = "cs_" + str(noOfCompositeStates)
                map_compositeState2hasStates[compositeStateName] = [state.name]
                map_state2compositeState[state.name].append(compositeStateName)
                allNewStates.append(compositeStateName)
                for constrainingState in constrainingStatesAnd:
                    map_compositeState2hasStates[compositeStateName].append(constrainingState.name + "_bar")
                G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name, agentTypeBar = True)
                node_labels[compositeStateName] = compositeStateName



        else:
            for i in range(len(constrainingStatesAnd)):
                listOfAgents =  []
                listOfAgents.append(state.correspondingAgent[0].name)
                noOfCompositeStates += 1
                compositeStateName = "cs_" + str(noOfCompositeStates)
                map_compositeState2hasStates[compositeStateName] = [state.name] 
                map_state2compositeState[state.name].append(compositeStateName)
                allNewStates.append(compositeStateName)
                map_compositeState2hasStates[compositeStateName].append(constrainingStatesAnd[i].name + "_bar")
                listOfAgents.append(constrainingStatesAnd[i].correspondingAgent[0].name)
                G.add_node(compositeStateName, type="composite state", agents = listOfAgents, baseAgent = state.correspondingAgent[0].name,agentTypeBar = True)
                node_labels[compositeStateName] = compositeStateName

    return noOfCompositeStates, map_compositeState2hasStates, map_state2compositeState, G, node_labels, allNewStates

def areTwoCompositeStatesAreConflictFree(compositeState1, compositeState2, map_compositeState2hasStates, G, onto):
    # Add an edge between the composite states only if 
    hasStates1 = map_compositeState2hasStates[compositeState1]
    hasStates2 = map_compositeState2hasStates[compositeState2]
    
    agentsCS1 = G.nodes[compositeState1]['agents']
    agentsCS2 = G.nodes[compositeState2]['agents']

    baseAgent1 = G.nodes[compositeState1]['baseAgent']
    baseAgent2 = G.nodes[compositeState2]['baseAgent']
    
    nominalAction = False
    if baseAgent1 == baseAgent2:
        nominalAction = True

    hasAtleastOneCommonAgent = False
    allCommonAgentsHaveSameState = False
    # Find a list of all same agents and their corresponding states
    sameAgentStates = []
    for agent1 in agentsCS1:
        if nominalAction:
            if baseAgent1 == agent1:
                continue
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
    
    if not hasAtleastOneCommonAgent:
        return True
    return allCommonAgentsHaveSameState
        
def checkIfOutgoingEdgesHaveTimePassageAction(G, node):
    # check if the cs has any outgoing edge of timePassageAction
    allOutgoingEdges = G.out_edges(node)
    timePassageActionExists = False
    for edge in allOutgoingEdges:
        if G.get_edge_data(edge[0], edge[1])['action'] == "timePassageAction":
            timePassageActionExists = True
            return True
    return False

def add_nomianlActionEdgesforAgentIndividual(allStates, map_state2compositeState, map_compositeState2hasStates , possibleNominalActions, G, edge_labels, edge_actingAgent, agentName, onto):
    for state in allStates:
        
        for idxComp1 in range(0,len(map_state2compositeState[state.name])): 
            # First composite
            cs1 = map_state2compositeState[state.name][idxComp1]
            if G.nodes[cs1]['agentTypeBar']:
                continue
            for action in possibleNominalActions:
                # Check if state can perform the action
                st_reachedByAction = getattr(state, action)
                if len(st_reachedByAction) == 0:
                    continue

                # Find all the composite states that are the children of the states that can be reachedByAction and define nominal reachability between them
                for reachableState in st_reachedByAction:
                    # Corresponding composite state
                    allCompositeStates = map_state2compositeState[reachableState.name]
                    for compositeState in allCompositeStates:
                        # Check if the composite state is free of conflicts for hasStates of other agents
                        bothNodesConflictFree = areTwoCompositeStatesAreConflictFree(cs1, compositeState, map_compositeState2hasStates, G, onto)
                        
                        if bothNodesConflictFree:
                        # Define the edge between the nodes in the graph with a cost
                            if action == "timePassageAction":
                                costOfTraversal = 1
                            else:
                                costOfTraversal = 10
                            # check if the cs has any outgoing edge of timePassageAction
                            cs1HasTimePassageAction = checkIfOutgoingEdgesHaveTimePassageAction(G, cs1)
                            if cs1HasTimePassageAction:
                                # Then only add new edge if it is timepassage action as well
                                if action != "timePassageAction":
                                    continue
                            G.add_edge(cs1, compositeState, action = action, cost = costOfTraversal, actingAgent = agentName)
                            edge_labels[(cs1, compositeState)] = action
                            edge_actingAgent[(cs1, compositeState)] = agentName

    return G, edge_labels, edge_actingAgent



def is_conflict_free(current_states, nodeHasStates, nodeAgentList,onto, actingAgent):
    # Return True if the action does not create a conflict, otherwise return False

    # The current states can be in bar states
    # Checking if the next_node composite state has any states either bar or not bar states which are in conflict with the current states

    allStatesInNextNodeConflictFree = True
    
    for i in range(0,len(nodeHasStates)):
        # one of hasState in the next node 
        stateOfInt = nodeHasStates[i]
        agentOfStateOfInt = nodeAgentList[i]
        stateOfAgentInCurrentState = current_states[agentOfStateOfInt]
        if agentOfStateOfInt == actingAgent:
            continue
        currentStateInBar = False
        # check if stateOgAgentInCurrentState is a bar state
        if "_bar" in stateOfAgentInCurrentState:
            # Lets find state without the bar:
            stateOfAgentInCurrentState_withoutBar = stateOfAgentInCurrentState[:-4]
            setOfStatesOfAgent_current = set(get_corresponding_iri(onto,agentOfStateOfInt).corrospondedBy)
            # remove the barOfState2 from the set
            bar_setOfStatesOfAgent_current = setOfStatesOfAgent_current - {get_corresponding_iri(onto,stateOfAgentInCurrentState_withoutBar)}
            currentStateInBar = True

        nextStateInBar = False
        if "_bar" in stateOfInt:
            # Lets find state without the bar:
            stateOfInt_withoutBar = stateOfInt[:-4]
            setOfStatesOfAgent = set(get_corresponding_iri(onto,agentOfStateOfInt).corrospondedBy)
            # remove the barOfState2 from the set
            bar_setOfStatesOfAgent = setOfStatesOfAgent - {get_corresponding_iri(onto,stateOfInt_withoutBar)}
            nextStateInBar = True

        if (not currentStateInBar and not nextStateInBar) or (currentStateInBar and nextStateInBar):
            if stateOfAgentInCurrentState == stateOfInt:
                allStatesInNextNodeConflictFree = True
            else:
                allStatesInNextNodeConflictFree = False
                return allStatesInNextNodeConflictFree
            
        elif not currentStateInBar and nextStateInBar:
            if get_corresponding_iri(onto, stateOfAgentInCurrentState) in bar_setOfStatesOfAgent: 
                allStatesInNextNodeConflictFree = True
            else:
                allStatesInNextNodeConflictFree = False
                break
        
        elif currentStateInBar and not nextStateInBar:
            if get_corresponding_iri(onto,stateOfInt) in bar_setOfStatesOfAgent_current:
                allStatesInNextNodeConflictFree = True
            else:
                allStatesInNextNodeConflictFree = False
                break 
            
    return allStatesInNextNodeConflictFree


def modifyCurrentStatesBasedOnAction(current_states, nextNodeHasStates, nextNodeAgentList):
    # Implement your logic to update the current states based on the action
    # Return the updated current states
    new_states = deepcopy(current_states)
    for i in range(0,len(nextNodeHasStates)):
        stateOfInt = nextNodeHasStates[i]
        agentOfStateOfInt = nextNodeAgentList[i]
        new_states[agentOfStateOfInt] = stateOfInt
    
    return new_states

def modifyCurrentStatesBasedOnActionV2(current_states, nextNodeHasStates, nextNodeAgentList, edgeActingAgent):
    # Implement your logic to update the current states based on the action
    # Return the updated current states
    new_states = deepcopy(current_states)
    for i in range(0,len(nextNodeHasStates)):
        stateOfInt = nextNodeHasStates[i]
        agentOfStateOfInt = nextNodeAgentList[i]
        if agentOfStateOfInt == edgeActingAgent:
            new_states[agentOfStateOfInt] = stateOfInt
    
    return new_states

def getSetOfNotBarState(onto, state):
    agentOfStateOfInt = get_corresponding_iri(onto,state).correspondingAgent[0]
    setOfStatesOfAgent = set(agentOfStateOfInt.corrospondedBy)
    # remove the barOfState2 from the set
    bar_setOfStatesOfAgent = setOfStatesOfAgent - {get_corresponding_iri(onto,state)}
    return bar_setOfStatesOfAgent


def checkIf2FullStatesAreSame(fullState1, fullState2, onto):
    # Check if the two full states are same
    for agent in fullState1.keys():
        fullState1_state = fullState1[agent]
        fullState2_state = fullState2[agent]
        if fullState1_state == fullState2_state:
            continue
        else:
            # Check if either of them is a bar state
            if "_bar" not in fullState1_state and "_bar" not in fullState2_state:
                return False
            else:
                st1_hasBar = "_bar" in fullState1_state
                if st1_hasBar:
                    nonBarState1 = fullState1_state[:-4]
                    setOfNonBarSate1 = getSetOfNotBarState(onto, nonBarState1)
                else:
                    setOfNonBarSate1 = set()
                    setOfNonBarSate1.add(get_corresponding_iri(onto,fullState1_state))

                st2_hasBar = "_bar" in fullState2_state
                if st2_hasBar:
                    nonBarState2 = fullState2_state[:-4]
                    setOfNonBarSate2 = getSetOfNotBarState(onto, nonBarState2)
                else:
                    setOfNonBarSate2 = set()
                    setOfNonBarSate2.add(get_corresponding_iri(onto,fullState2_state))
                
                if st1_hasBar and st2_hasBar:
                    if setOfNonBarSate1 == (setOfNonBarSate2) :
                        continue
                    else:
                        return False
                elif st1_hasBar and not st2_hasBar:
                    if setOfNonBarSate2.issubset(setOfNonBarSate1):
                        continue
                    else:
                        return False
                elif not st1_hasBar and st2_hasBar:
                    if setOfNonBarSate1.issubset(setOfNonBarSate2):
                        continue
                    else:
                        return False       
    return True

def getSortedNeighborsBasedOnCost(G, current_node):
    # Get all neighbors of the current node and sort them based on the cost
    neighbors = G.neighbors(current_node)
    sortedNeighbors = []
    for neighbor in neighbors:
        edgeData = G.get_edge_data( current_node , neighbor )
        sortedNeighbors.append((neighbor, edgeData['cost']))
    sortedNeighbors.sort(key=lambda x: x[1])
    return sortedNeighbors


def visSankeyGraphActions(actions):
    # Generate states as placeholders
    states = [f"State{i}" for i in range(len(actions) + 1)]

    # Create flows between states based on actions
    source = states[:-1]  # From State0 to State(n-1)
    target = states[1:]   # From State1 to State(n)
    edges = list(zip(source, target, actions))

    uniqueActions = []
    countList = []
    uniqueActions.append(actions[0])
    oldAction = actions[0]
    count = 1
    for idx in range(1, len(actions)):
        currAction = actions[idx]
        if "timePassageAction"  in currAction:
            continue
        if currAction == oldAction:
            count = count + 1
        else:
            countList.append(count)
            uniqueActions.append(currAction)
            count = 1
            oldAction = currAction
    countList.append(count)

    # Count consecutive actions to determine flow thickness
    # find if the previous action is same as the current action
    statesUQ = [f"State{i}" for i in range(len(uniqueActions) + 1)]

    # create a graph with the statesUQ in the nodes and uniqueActions in the edges. The thickness of the edge depends on the countList
    visualize_graph_with_actions(statesUQ, uniqueActions, countList)


def visualize_graph_with_actions(statesUQ, uniqueActions, countList):
    G = nx.DiGraph()

    # Add nodes for each state
    for state in statesUQ:
        G.add_node(state)

    # Add edges for each action with thickness based on countList
    for i, action in enumerate(uniqueActions):
        source_state = statesUQ[i]
        target_state = statesUQ[i + 1]
        edge_thickness = countList[i]
        G.add_edge(source_state, target_state, label=action, weight=edge_thickness)

    pos = nx.spring_layout(G)  # positions for all nodes

    # Create edge traces
    edge_traces = []
    edge_labels = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge[2]['weight'], color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
        edge_labels.append((x0 + x1) / 2)
        edge_labels.append((y0 + y1) / 2)
        edge_labels.append(edge[2]['label'])

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

    # Create edge label traces
    edge_label_trace = go.Scatter(
        x=edge_labels[0::3],
        y=edge_labels[1::3],
        text=edge_labels[2::3],
        mode='text',
        hoverinfo='none',
        textfont=dict(size=16, family='Arial Bold', weight='bold')  # Increase font size and make text bold
        # textfont=dict(size=16) 
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace, edge_label_trace],
                    layout=go.Layout(
                        title='Action Sequence Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Show the interactive plot
    fig.show()


def printOnlyActions(listOfActions):
    
    condensedList = []
    for action in listOfActions:
        if "call" not in action:
            print(action) 
            condensedList.append(action)
    visSankeyGraphActions(condensedList)
    return condensedList

def find_all_paths_with_conflict_check_DFS(G, start_node, end_node,map_compositeState2hasStates, current_states,onto, FS, agentWhichNeedsToBeControlled):
    def dfs(current_node, path, current_states, map_compositeState2hasStates,onto, visited, current_cost, actionsList, fullStateList):
        
        # global start_timeInitial  # Declare the global variable

        if current_states[agentWhichNeedsToBeControlled] == FS[agentWhichNeedsToBeControlled]:
           print("Found Something")    
        #    print("Action: ", actionsList)
           justListOfAction = printOnlyActions(actionsList)
           print("Action: ", justListOfAction)
           total_time = time.time() - globals.start_timeInitial
           print("Total Time: ", total_time)
           sys.exit()

        if current_node == end_node:
            all_paths.append((path, current_cost, actionsList, fullStateList))
            # return
        visited.append(current_node)
        print("Current Node: ", current_node, "has states: ", map_compositeState2hasStates[current_node], "Agent States: ", G.nodes[current_node]['baseAgent'])

        listOfSortedNeighbors = getSortedNeighborsBasedOnCost(G, current_node)
        for neighbor, costToGo in listOfSortedNeighbors: #G.neighbors(current_node):
            allowForGoingBackToSameNode = True
            bothStatesSame = False
            if allowForGoingBackToSameNode:
                if neighbor in visited:
                    # Check if the last time the neighbor was visited the full states were same or different
                    # Get all indices of the element
                    for index, value in enumerate(path) :
                        if value == neighbor:
                            idxNeighbor = index
                            fullStateNeighbor_lastVisit = fullStateList[idxNeighbor]
                            fullStateNeighbor_thisVisit = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                            bothStatesSame = checkIf2FullStatesAreSame(fullStateNeighbor_lastVisit, fullStateNeighbor_thisVisit, onto)
                            if bothStatesSame:
                                break   
            else:
                bothStatesSame = neighbor in visited
            if not bothStatesSame:
                edgeData = G.get_edge_data( current_node , neighbor )
                edge_cost = edgeData['cost'] 
                edge_action = edgeData['action']
                edge_actingAgent = edgeData['actingAgent']

                current2neighborIsConflictFree = True
                current2neighborIsConflictFree = is_conflict_free(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'],onto, edgeData['actingAgent'] )
                
                if current2neighborIsConflictFree:
                    if edge_cost < 100:
                        # new_states = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                        new_states = modifyCurrentStatesBasedOnActionV2(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'], edge_actingAgent)
                        
                    # new_states = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                    else:
                        new_states = deepcopy(current_states)
                    node_data = G.nodes[neighbor]
                    baseStateOfNeighbor = node_data['baseAgent']
                    if edge_action == "callDifferentAgent":
                        edge_action = "call Agent "+  baseStateOfNeighbor
                    else:
                        edge_action = edge_action + " by agent " + baseStateOfNeighbor
                    print("Going to neighbor: ", neighbor, " with action: ", edge_action)
                    dfs(neighbor, path + [neighbor], new_states, map_compositeState2hasStates,onto, visited, current_cost + edge_cost, actionsList + [edge_action], fullStateList + [deepcopy(new_states)])
        print("No more neighbors to visit, going a step back ")
        # visited.remove(current_node)

    all_paths = []
    actual_fullState = deepcopy(current_states)
    # Start Node, Path, Current States, Map of composite state to hasStates, Ontology, visited list, cost, actionsList, FullState
    dfs(start_node, [start_node], current_states, map_compositeState2hasStates, onto, [], 0, [], [current_states])
    return all_paths


def find_all_paths_with_conflict_check_DFS_V1(G, start_node, end_node,map_compositeState2hasStates, current_states,onto, FS, agentWhichNeedsToBeControlled):
    def dfs(current_node, path, current_states, map_compositeState2hasStates,onto, visited, current_cost, actionsList, fullStateList):
        
        if current_states[agentWhichNeedsToBeControlled] == FS[agentWhichNeedsToBeControlled]:
           print("Found Something")    
        if current_node == end_node:
            all_paths.append((path, current_cost, actionsList, fullStateList))
            return
        visited.append(current_node)
        listOfSortedNeighbors = getSortedNeighborsBasedOnCost(G, current_node)
        for neighbor, costToGo in listOfSortedNeighbors: #G.neighbors(current_node):
            allowForGoingBackToSameNode = False
            bothStatesSame = False
            if allowForGoingBackToSameNode:
                if neighbor in visited:
                    # Check if the last time the neighbor was visited the full states were same or different
                    # Get all indices of the element
                    for index, value in enumerate(path) :
                        if value == neighbor:
                            idxNeighbor = index
                            fullStateNeighbor_lastVisit = fullStateList[idxNeighbor]
                            fullStateNeighbor_thisVisit = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                            bothStatesSame = checkIf2FullStatesAreSame(fullStateNeighbor_lastVisit, fullStateNeighbor_thisVisit, onto)
                            if bothStatesSame:
                                break   
            else:
                bothStatesSame = neighbor in visited
            if not bothStatesSame:
                edgeData = G.get_edge_data( current_node , neighbor )
                edge_cost = edgeData['cost'] 
                edge_action = edgeData['action']

                current2neighborIsConflictFree = True
                current2neighborIsConflictFree = is_conflict_free(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'],onto, edgeData['actingAgent'] )
                
                if current2neighborIsConflictFree:
                    new_states = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                    node_data = G.nodes[neighbor]
                    baseStateOfNeighbor = node_data['baseAgent']
                    if edge_action == "callDifferentAgent":
                        edge_action = "call Agent "+  baseStateOfNeighbor
                    else:
                        edge_action = edge_action + " by agent " + baseStateOfNeighbor

                    dfs(neighbor, path + [neighbor], new_states, map_compositeState2hasStates,onto, visited, current_cost + edge_cost, actionsList + [edge_action], fullStateList + [deepcopy(new_states)])

        visited.remove(current_node)

    all_paths = []
    actual_fullState = deepcopy(current_states)
    # Start Node, Path, Current States, Map of composite state to hasStates, Ontology, visited list, cost, actionsList, FullState
    dfs(start_node, [start_node], current_states, map_compositeState2hasStates, onto, [], 0, [], [current_states])
    return all_paths
    
def inFullStateNode(listOfAllFullStateNode , currentState):
    for len in range(0,len(listOfAllFullStateNode)):
        nodeIdx = len
        nodeDef = listOfAllFullStateNode[nodeIdx]
        if nodeDef == currentState:
            return nodeIdx, listOfAllFullStateNode
    return -1

def pairInNodesToExplore(nodes_to_explore, pair_fullStateCompositeNode, onto):
    for i in range(0,len(nodes_to_explore)):
        if checkIf2FullStatesAreSame (nodes_to_explore[i][2], pair_fullStateCompositeNode[0], onto) and nodes_to_explore[i][3] == pair_fullStateCompositeNode[1]:
            return i
    return -1

def pairInExpanded(expanded, pair_fullStateCompositeNode, onto):
    for i in range(0,len(expanded)):
        fullStateNode_e = expanded[i][0]
        cn_e = expanded[i][1]
        if checkIf2FullStatesAreSame(pair_fullStateCompositeNode[0], fullStateNode_e, onto) and pair_fullStateCompositeNode[1] == cn_e:
            return i
    return -1

def find_all_paths_with_conflict_check_Dijkstras(G, start_node, end_node, map_compositeState2hasStates, current_states, onto,  FS, agentWhichNeedsToBeControlled):
    def dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto, FS, agentWhichNeedsToBeControlled):
        
        list_allFullStateNodeCompositeStatePair.append((deepcopy(current_states), start_node,0))
        idx = 0

        # Cost, Current Node Idx in list_allFullStateNodes, Current Node, corresponding composite node in actual graph, Path, ActionsList
        nodes_to_explore = [(0, idx, list_allFullStateNodeCompositeStatePair[idx][0], list_allFullStateNodeCompositeStatePair[idx][1], [idx], [])]
        expanded = []
        while nodes_to_explore:
            # Sort and find the node with the smallest cost
            nodes_to_explore.sort(key=lambda x: x[0])
            current_cost, currentNodeIdx ,current_node, nodeInGraph, path, actionsList = nodes_to_explore.pop(0)
            print("Current Node: ", nodeInGraph)
            if current_node[agentWhichNeedsToBeControlled] == FS[agentWhichNeedsToBeControlled]:
                all_paths.append((path, current_cost, actionsList))
                continue

            fullStateNodeCompositeStatePair = list_allFullStateNodeCompositeStatePair[currentNodeIdx]

            nodeIdxInExpanded = pairInExpanded(expanded, fullStateNodeCompositeStatePair, onto)
            
            if nodeIdxInExpanded >= 0:
                continue
            else:
                listOfSortedNeighbors = getSortedNeighborsBasedOnCost(G, nodeInGraph)
                for neighbor, edge_cost in listOfSortedNeighbors:
                    print("     Checking Neighbor: ", neighbor)
                    # if nodeInGraph == 'cs_4':
                    #     print("debug")
                    # if neighbor == 'cs_23':
                    #     print("debug")
                    # if nodeInGraph == 'cs_59':
                    #     print("debug")
                    # if neighbor == 'cs_107':
                    #     print("debug")
                    nextWouldBeFullState = modifyCurrentStatesBasedOnAction(current_node, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                    nextWouldBeCompositState = neighbor
                    edgeData = G.get_edge_data(nodeInGraph, neighbor)
                    edge_action = edgeData['action']
                    current2neighborIsConflictFree = is_conflict_free(current_node, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'], onto, edgeData['actingAgent'])
                    if current2neighborIsConflictFree:
                        new_cost = current_cost + edge_cost
                        idxOfNextPairInNodesToExplore = pairInNodesToExplore(nodes_to_explore, (nextWouldBeFullState, nextWouldBeCompositState), onto)
                        idxOfNextPairInExpanded = pairInExpanded(expanded, (nextWouldBeFullState, nextWouldBeCompositState), onto)
                        addToList = False
                        addToExpanded = False
                        if idxOfNextPairInNodesToExplore == -1:
                            # The pair is not in the list to explore
                            # This means it has already been expanded or the frontier had not reached it yet
                            print("         Neighbor not in nodes to explore")
                            if idxOfNextPairInExpanded == -1:
                                # The pair is not in the expanded list
                                # This means frontier just reached it
                                print("         Neighbor not in expanded")
                                addToList = True
                            else:
                                # The pair is in the expanded list
                                # This means there is a way to reach it. So if the cost is minimal we update it
                                print("         Neighbor in expanded")
                                oldCost = expanded[idxOfNextPairInExpanded][2]
                                if new_cost < oldCost:
                                    print("         Neighbor in expanded and new cost is less")
                                    addToList = True
                                    addToExpanded = True
                                    nodes_to_explore.pop(idxOfNextPairInNodesToExplore)
                                    # expanded.pop(idxOfNextPairInExpanded)
                                    expanded[idxOfNextPairInExpanded][1] = new_cost
                                else:
                                    print("         Neighbor in expanded but new cost is more")

                        else:
                            # The pair is in the list to explore
                            # This means it has not been expanded yet
                            # So we can check if the cost is minimal
                            print("         Neighbor in nodes to explore")
                            oldCost = nodes_to_explore[idxOfNextPairInNodesToExplore][0]
                            if new_cost < oldCost:
                                print("         Neighbor in nodes to explore and new cost is less")
                                nodes_to_explore.pop(idxOfNextPairInNodesToExplore)
                                addToList = True
                            else:
                                print("         Neighbor in nodes to explore but new cost is more")
                                
                        if addToList:
                                # Can add to the list
                                print("         Adding the pair in nodes to explore")
                                node_data = G.nodes[neighbor]
                                baseStateOfNeighbor = node_data['baseAgent']
                                if edge_action == "callDifferentAgent":
                                    edge_action = "call Agent " + baseStateOfNeighbor
                                else:
                                    edge_action = edge_action + " by agent " + baseStateOfNeighbor
                                
                                
                                list_allFullStateNodeCompositeStatePair.append((deepcopy(nextWouldBeFullState), deepcopy(neighbor), deepcopy(new_cost)))

                                # Cost, Current Node Idx in list_allFullStateNodes, Current Node, corresponding composite node in actual graph, Path, ActionsList
                                idx = len(list_allFullStateNodeCompositeStatePair)-1
                                nodes_to_explore.append((new_cost, idx, list_allFullStateNodeCompositeStatePair[idx][0], list_allFullStateNodeCompositeStatePair[idx][1], path + [neighbor], actionsList + [edge_action]))
                                print("             Neighbor of : ", nodeInGraph, "is  " , neighbor, "with action: ", edge_action)

                    else:
                        print("     Neighbor not added due to conflict between current and next full states are in conflict")
                expanded.append(fullStateNodeCompositeStatePair)
                print("Debug")    
                
    list_allFullStateNodeCompositeStatePair = []
    
    all_paths = []
    dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto, FS, agentWhichNeedsToBeControlled)
    return all_paths


def find_all_paths_with_conflict_check_DijkstrasV1(G, start_node, end_node, map_compositeState2hasStates, current_states, onto):
    def dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto):
        nodes_to_explore = [(0, start_node, [start_node], current_states, [], [deepcopy(current_states)])]
        visited = set()
        node_costs = {start_node: 0}  # Dictionary to keep track of the minimum cost to reach each node

        while nodes_to_explore:
            # Find the node with the smallest cost
            nodes_to_explore.sort(key=lambda x: x[0])
            current_cost, current_node, path, current_states, actionsList, fullStateList = nodes_to_explore.pop(0)
            
            if current_node == end_node:
                all_paths.append((path, current_cost, actionsList, fullStateList))
                continue
            
            if current_node in visited:
                bothStatesSame = False
                for index, value in enumerate(path):
                    if value == current_node:
                        idxNeighbor = index
                        fullStateNeighbor_lastVisit = fullStateList[idxNeighbor]
                        fullStateNeighbor_thisVisit = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[current_node], G.nodes[current_node]['agents'])
                        bothStatesSame = checkIf2FullStatesAreSame(fullStateNeighbor_lastVisit, fullStateNeighbor_thisVisit, onto)
                        if bothStatesSame:
                            break
                if bothStatesSame:
                    continue
            
            visited.add(current_node)
            listOfSortedNeighbors = getSortedNeighborsBasedOnCost(G, current_node)
            for neighbor, edge_cost in listOfSortedNeighbors:
                edgeData = G.get_edge_data(current_node, neighbor)
                edge_action = edgeData['action']
                current2neighborIsConflictFree = is_conflict_free(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'], onto, edgeData['actingAgent'])
                
                if current2neighborIsConflictFree:
                    new_cost = current_cost + edge_cost
                    if neighbor not in node_costs or new_cost < node_costs[neighbor]:
                        node_costs[neighbor] = new_cost
                        new_states = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
                        node_data = G.nodes[neighbor]
                        baseStateOfNeighbor = node_data['baseAgent']
                        if edge_action == "callDifferentAgent":
                            edge_action = "call Agent " + baseStateOfNeighbor
                        else:
                            edge_action = edge_action + " by agent " + baseStateOfNeighbor
                        nodes_to_explore.append((new_cost, neighbor, path + [neighbor], new_states, actionsList + [edge_action], fullStateList + [deepcopy(new_states)]))
                        print("Neighbor of : ", current_node, "is  " , neighbor)
                        print("sus")

    all_paths = []
    dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto)
    return all_paths



# def find_all_paths_with_conflict_check_Dijkstras(G, start_node, end_node, map_compositeState2hasStates, current_states, onto):
#     def dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto):
#         priority_queue = [(0, start_node, [start_node], current_states, [], [deepcopy(current_states)])]
#         visited = set()
#         while priority_queue:
#             current_cost, current_node, path, current_states, actionsList, fullStateList = heapq.heappop(priority_queue)
#             if current_node == end_node:
#                 all_paths.append((path, current_cost, actionsList, fullStateList))
#                 continue
#             if current_node in visited:
#                 bothStatesSame = False
#                 for index, value in enumerate(path) :
#                     if value == current_node:
#                         idxNeighbor = index
#                         fullStateNeighbor_lastVisit = fullStateList[idxNeighbor]
#                         fullStateNeighbor_thisVisit = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
#                         bothStatesSame = checkIf2FullStatesAreSame(fullStateNeighbor_lastVisit, fullStateNeighbor_thisVisit, onto)
#                         if bothStatesSame:
#                             break   
#                 if bothStatesSame:
#                     continue
#             visited.add(current_node)
#             listOfSortedNeighbors = getSortedNeighborsBasedOnCost(G, current_node)
#             for neighbor, edge_cost in listOfSortedNeighbors:
#                 edgeData = G.get_edge_data(current_node, neighbor)
#                 edge_action = edgeData['action']
#                 current2neighborIsConflictFree = is_conflict_free(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'], onto, edgeData['actingAgent'])
#                 if current2neighborIsConflictFree:
#                     new_states = modifyCurrentStatesBasedOnAction(current_states, map_compositeState2hasStates[neighbor], G.nodes[neighbor]['agents'])
#                     node_data = G.nodes[neighbor]
#                     baseStateOfNeighbor = node_data['baseAgent']
#                     if edge_action == "callDifferentAgent":
#                         edge_action = "call Agent " + baseStateOfNeighbor
#                     else:
#                         edge_action = edge_action + " by agent " + baseStateOfNeighbor
#                     heapq.heappush(priority_queue, (current_cost + edge_cost, neighbor, path + [neighbor], new_states, actionsList + [edge_action], fullStateList + [deepcopy(new_states)]))
#                     print("Neighbor of : ", current_node, "is  " , neighbor)
#                     outsy = 1
                
#     all_paths = []
#     dijkstra(start_node, end_node, current_states, map_compositeState2hasStates, onto)
#     return all_paths