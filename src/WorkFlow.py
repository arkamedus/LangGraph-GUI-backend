import os
import re
import json
from typing import Dict, List, TypedDict, Any, Annotated, Callable, Literal, Optional, Union
import operator
import inspect

from langgraph.graph import StateGraph, END, START

from NodeData import NodeData
from llm import get_llm, clip_history, create_llm_chain
from util import flush_print, with_metadata

# Tool registry
tool_registry: Dict[str, Callable] = {}
tool_info_registry: Dict[str, str] = {}

# Subgraph registry
subgraph_registry: Dict[str, Any] = {}

# Decorator to register tools
def tool(func: Callable) -> Callable:
    signature = inspect.signature(func)
    docstring = func.__doc__ or ""
    tool_info = f"{func.__name__}{signature} - {docstring}"
    tool_registry[func.__name__] = func
    tool_info_registry[func.__name__] = tool_info
    return func

def parse_nodes_from_json(graph_data: Dict[str, Any]) -> Dict[str, NodeData]:
    node_map = {}
    for node_data in graph_data.get("nodes", []):
        node = NodeData.from_dict(node_data)
        node_map[node.uniq_id] = node
    return node_map

def find_nodes_by_type(node_map: Dict[str, NodeData], node_type: str) -> List[NodeData]:
    return [node for node in node_map.values() if node.type == node_type]


class PipelineState(TypedDict):
    history: Annotated[str, operator.add]
    task: Annotated[str, operator.add]
    condition: Annotated[bool, lambda x, y: y]

@with_metadata
def execute_step(state: PipelineState, name: str, prompt_template: str, llm, **metadata) -> PipelineState:
    flush_print(f"{name} (ID: {metadata['node_id']}) is working...")
    state["history"] = clip_history(state["history"])
    data = json.loads(create_llm_chain(prompt_template, llm, state["history"]))
    state["history"] += "\n" + json.dumps(data)
    state["history"] = clip_history(state["history"])
    flush_print(state["history"])
    return state

@with_metadata
def execute_tool(state: PipelineState, name: str, prompt_template: str, llm, **metadata) -> PipelineState:
    flush_print(f"{name} (ID: {metadata['node_id']}) is working...")
    state["history"] = clip_history(state["history"])
    generation = create_llm_chain(prompt_template, llm, state["history"])
    sanitized_generation = re.sub(r'[\x00-\x1F\x7F]', '', generation)
    flush_print(sanitized_generation)
    data = json.loads(sanitized_generation)

    tool_name, args = data["function"], data["args"]
    if tool_name not in tool_registry:
        raise ValueError(f"Tool {tool_name} not found in registry.")

    result = tool_registry[tool_name](*args)
    flush_print(f"Executed Tool: {tool_name}({', '.join(map(str, args))}) Result: {result}")

    state["history"] += f"\nExecuted {tool_name}({', '.join(map(str, args))}) Result: {result}"
    state["history"] = clip_history(state["history"])
    return state

@with_metadata
def condition_switch(state: PipelineState, name: str, prompt_template: str, llm, **metadata) -> PipelineState:
    flush_print(f"{name} (ID: {metadata['node_id']}) is working...")
    state["history"] = clip_history(state["history"])
    data = json.loads(create_llm_chain(prompt_template, llm, state["history"]))
    state["condition"] = data["switch"]
    flush_print(f"Condition is {state['condition']}")
    return state


def info_add(name: str, state: PipelineState, information: str, llm) -> PipelineState:
    flush_print(f"{name} is adding information...")
    state["history"] += "\n" + information
    state["history"] = clip_history(state["history"])
    return state


def sg_add(name: str, state: PipelineState, sg_name: str) -> PipelineState:
    flush_print(f"{name} is working, it is a subgraph node call {sg_name} ...")
    subgraph = subgraph_registry[sg_name]
    response = subgraph.invoke(
        PipelineState(
            history=state["history"],
            task=state["task"],
            condition=state["condition"]
        )
    )
    state["history"] = response["history"]
    state["task"] = response["task"]
    state["condition"] = response["condition"]
    return state


def conditional_edge(state: PipelineState) -> Literal["True", "False"]:
    return "True" if state["condition"] in ["True", "true", True] else "False"


def build_subgraph(node_map: Dict[str, NodeData], llm) -> StateGraph:
    subgraph = StateGraph(PipelineState)

    # Ensure START node is added first
    start_node = find_nodes_by_type(node_map, "START")[0]
    flush_print(f"Start root ID: {start_node.uniq_id}")

    subgraph.add_node(start_node.uniq_id, lambda state: state)  # No-op start node

    # Step nodes
    for node in find_nodes_by_type(node_map, "STEP"):
        prompt_template = f"""
        history: {{history}}
        {node.description}
        you reply in the json format
        """
        subgraph.add_node(
            node.uniq_id,
            lambda state, template=prompt_template, llm=llm, node_id=node.uniq_id, name=node.name:
            execute_step(state, name, template, llm, graph="workflow", subgraph="workflow", node_id=node_id, node_type="STEP")
        )

    # INFO nodes
    for node in find_nodes_by_type(node_map, "INFO"):
        subgraph.add_node(
            node.uniq_id,
            lambda state, template=node.description, llm=llm, name=node.name: info_add(name, state, template, llm)
        )

    # SUBGRAPH nodes
    for node in find_nodes_by_type(node_map, "SUBGRAPH"):
        subgraph.add_node(
            node.uniq_id,
            lambda state, llm=llm, name=node.name, sg_name=node.name: sg_add(name, state, sg_name)
        )

    # Ensure all nodes are added before defining edges
    for node in node_map.values():
        for next_id in node.nexts:
            subgraph.add_edge(node.uniq_id, next_id)

    # Conditions
    for node in find_nodes_by_type(node_map, "CONDITION"):
        subgraph.add_node(
            node.uniq_id,
            lambda state, template=node.description, llm=llm, name=node.name:
            condition_switch(state, name, template, llm, graph="workflow", subgraph="workflow", node_id=node.uniq_id, node_type="CONDITION")
        )
        subgraph.add_conditional_edges(
            node.uniq_id, conditional_edge, {
                "True": node.true_next if node.true_next else END,
                "False": node.false_next if node.false_next else END
            }
        )

    return subgraph.compile()



class MainGraphState(TypedDict):
    input: Union[str, None]

def invoke_root(state: MainGraphState):
    subgraph = subgraph_registry["root"]
    response = subgraph.invoke(
        PipelineState(history="", task="", condition=False)
    )
    return {"input": None}


def run_workflow_as_server(llm):
    with open("graph.json", 'r') as file:
        graphs = json.load(file)

    for graph in graphs:
        subgraph_name = graph.get("name")
        node_map = parse_nodes_from_json(graph)

        for tool_node in find_nodes_by_type(node_map, "TOOL"):
            exec(tool_node.description, globals())

        subgraph = build_subgraph(node_map, llm)
        subgraph_registry[subgraph_name] = subgraph

    main_graph = StateGraph(MainGraphState)
    main_graph.add_node("subgraph", invoke_root)
    main_graph.set_entry_point("subgraph")
    main_graph = main_graph.compile()

    for state in main_graph.stream({"input": None}):
        flush_print(state)
