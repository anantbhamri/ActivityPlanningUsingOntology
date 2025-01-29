import os
from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt
import ollama
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

# My OpenAI API key
secret_openAI_apiKEY = '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'


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


# Function to find the shortest path using Dijkstra's algorithm
def find_shortest_path(graph, start_node, end_node):
    try:
        path = nx.dijkstra_path(graph, start_node, end_node)
        print(f"Shortest path from {start_node} to {end_node}: {path}")
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}")

# Function to find all paths between two nodes
def find_all_paths(graph, start_node, end_node):
    try:
        paths = list(nx.all_simple_paths(graph, start_node, end_node))
        if paths:
            print(f"All paths from {start_node} to {end_node}:")
            for path in paths:
                print(path)
        else:
            print(f"No paths found between {start_node} and {end_node}")
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}")



# Load the ontology
onto = get_ontology("/Users/bhamri/Desktop/Ontology/owlFiles/sm_v3_navigationRules.owl").load()

# Create an empty graph
G = nx.Graph()

# Add individuals as nodes
for individual in onto.individuals():
    G.add_node(individual.name, type="individual")

# Add relationships between individuals as edges
edge_labels = {}
for individual in onto.individuals():
    for prop in individual.get_properties():
        for value in prop[individual]:
            if isinstance(value, Thing):  # Check if the value is another individual
                G.add_edge(individual.name, value.name, relationship=prop.name)
                edge_labels[(individual.name, value.name)] = prop.name
# Print some basic information about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Read and convert SWRL rules to Python conditions
print("SWRL Rules:")
swrl_rules = ""
for rule in onto.rules():
    print(rule)
    # result = ask_ollama(f"Convert the following SWRL rules to a Python condition:\n\n{rule}\n\nPython condition:")
    # print(result)
    swrl_rules += str(rule) + "\n"


# python_condition = convert_swrl_to_python(swrl_rules)
# print(f"Python condition: {python_condition}")

# Example usage of the function
start_node = "ex1_PO"  # Replace with actual node name
end_node = "ex1_Barn"    # Replace with actual node name

# find_shortest_path(G, start_node, end_node)
# find_all_paths(G, start_node, end_node)

# Visualize the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',font_size=5)

plt.title("Ontology Graph")
plt.show()
