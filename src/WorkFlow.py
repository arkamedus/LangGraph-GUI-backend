# WorkFlow.py

import os
import re
import json
from typing import Dict, List, TypedDict, Any, Annotated, Callable, Literal, Union, Optional
import operator
import inspect

from langgraph.graph import StateGraph, END, START

from NodeData import NodeData
from llm import get_llm, clip_history, create_llm_chain

# Global metadata for logging
CURRENT_METADATA: Dict[str, Any] = {}

def flush_print(message: str, status: Optional[bool] = None):
    log = {
        "graph": CURRENT_METADATA.get("graph"),
        "subgraph": CURRENT_METADATA.get("subgraph"),
        "node": CURRENT_METADATA.get("node"),
        "node_id": CURRENT_METADATA.get("node_id"),
        "node_type": CURRENT_METADATA.get("node_type"),
        "status": status,
        "message": message
    }
    print(json.dumps(log), flush=True)

def with_metadata(fn: Callable, sg_name: str, node_name: str, node_id: str, node_type: str):
    def wrapped(state, *args, **kwargs):
        global CURRENT_METADATA
        CURRENT_METADATA = {
            "graph": sg_name,
            "subgraph": sg_name,
            "node": node_name,
            "node_id": node_id,
            "node_type": node_type
        }
        flush_print(f"START execution", status=True)
        result = fn(state, *args, **kwargs)
        flush_print(f"END execution", status=False)
        return result
    return wrapped

# Tool registry to hold information about tools
tool_registry: Dict[str, Callable] = {}
tool_info_registry: Dict[str, str] = {}

# Subgraph registry to hold all the subgraph
subgraph_registry: Dict[str, Any] = {}

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

def execute_step(name: str, state: PipelineState, prompt_template: str, llm) -> PipelineState:
    flush_print(f"{name} is working...", status=True)
    state["history"] = clip_history(state["history"])
    generation = create_llm_chain(prompt_template, llm, state["history"])
    data = json.loads(generation)
    state["history"] += "\n" + json.dumps(data)
    state["history"] = clip_history(state["history"])
    flush_print(state["history"], status=False)
    return state

def execute_tool(name: str, state: PipelineState, prompt_template: str, llm) -> PipelineState:
    flush_print(f"{name} is working...", status=True)
    state["history"] = clip_history(state["history"])
    generation = create_llm_chain(prompt_template, llm, state["history"])
    sanitized_generation = re.sub(r'[\x00-\x1F\x7F]', '', generation)
    flush_print(sanitized_generation, status=True)
    data = json.loads(sanitized_generation)
    tool_name = data["function"]
    args = data["args"]
    if tool_name not in tool_registry:
        raise ValueError(f"Tool {tool_name} not found in registry.")
    result = tool_registry[tool_name](*args)
    flattened_args = ', '.join(map(str, args))
    flush_print(f"Executed Tool: {tool_name}({flattened_args})  Result is: {result}", status=False)
    state["history"] += f"\nExecuted {tool_name}({flattened_args})  Result is: {result}"
    state["history"] = clip_history(state["history"])
    return state

def condition_switch(name: str, state: PipelineState, prompt_template: str, llm) -> PipelineState:
    flush_print(f"{name} is working...", status=True)
    state["history"] = clip_history(state["history"])
    generation = create_llm_chain(prompt_template, llm, state["history"])
    data = json.loads(generation)
    state["condition"] = data["switch"]
    state["history"] += f"\nCondition is {state['condition']}"
    state["history"] = clip_history(state["history"])
    return state

def info_add(name: str, state: PipelineState, information: str, llm) -> PipelineState:
    flush_print(f"{name} is adding information...", status=True)
    state["history"] += "\n" + information
    state["history"] = clip_history(state["history"])
    return state

def sg_add(name: str, state: PipelineState, sg_name: str) -> PipelineState:
    flush_print(f"{name} is working, it is a subgraph node call {sg_name} ...", status=True)
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
    return "True" if state["condition"] in [True, "True", "true"] else "False"

def build_subgraph(node_map: Dict[str, NodeData], llm, sg_name: str) -> StateGraph:
    subgraph = StateGraph(PipelineState)
    start_node = find_nodes_by_type(node_map, "START")[0]
    flush_print(f"Start root ID: {start_node.uniq_id}", status=True)

    step_nodes = find_nodes_by_type(node_map, "STEP")
    for current_node in step_nodes:
        if current_node.tool:
            tool_info = tool_info_registry[current_node.tool]
            prompt_template = f"""
            history: {{history}}
            {current_node.description}
            Available tool: {tool_info}
            Based on Available tool, arguments in the json format:
            "function": "<func_name>", "args": [<arg1>, <arg2>, ...]
            """
            node_fn = with_metadata(
                lambda state, template=prompt_template, llm=llm, name=current_node.name: execute_tool(name, state, template, llm),
                sg_name, current_node.name, current_node.uniq_id, current_node.type
            )
            subgraph.add_node(current_node.uniq_id, node_fn)
        else:
            prompt_template = f"""
            history: {{history}}
            {current_node.description}
            you reply in the json format
            """
            node_fn = with_metadata(
                lambda state, template=prompt_template, llm=llm, name=current_node.name: execute_step(name, state, template, llm),
                sg_name, current_node.name, current_node.uniq_id, current_node.type
            )
            subgraph.add_node(current_node.uniq_id, node_fn)

    info_nodes = find_nodes_by_type(node_map, "INFO")
    for info_node in info_nodes:
        node_fn = with_metadata(
            lambda state, template=info_node.description, llm=llm, name=info_node.name: info_add(name, state, template, llm),
            sg_name, info_node.name, info_node.uniq_id, info_node.type
        )
        subgraph.add_node(info_node.uniq_id, node_fn)

    subgraph_nodes = find_nodes_by_type(node_map, "SUBGRAPH")
    for sg_node in subgraph_nodes:
        node_fn = with_metadata(
            lambda state, llm=llm, name=sg_node.name, sg_name=sg_node.name: sg_add(name, state, sg_name),
        )
    # Edges
    # Find all next nodes from start_node
    next_node_ids = start_node.nexts
    next_nodes = [node_map[next_id] for next_id in next_node_ids]

    for next_node in next_nodes:
        flush_print(f"Next node ID: {next_node.uniq_id}, Type: {next_node.type}", status=True)
        subgraph.add_edge(START, next_node.uniq_id)

    for node in step_nodes + info_nodes + subgraph_nodes:
        next_nodes = [node_map[next_id] for next_id in node.nexts]

        for next_node in next_nodes:
            flush_print(f"{node.name} {node.uniq_id}'s next node: {next_node.name} {next_node.uniq_id}, Type: {next_node.type}", status=True)
            subgraph.add_edge(node.uniq_id, next_node.uniq_id)

    condition_nodes = find_nodes_by_type(node_map, "CONDITION")
    for condition in condition_nodes:
        condition_template = f"""{condition.description}
        history: {{history}}, decide the condition result in the json format:
        "switch": True/False
        """
        node_fn = with_metadata(
            lambda state, template=condition_template, llm=llm, name=condition.name: condition_switch(name, state, template, llm),
            sg_name, condition.name, condition.uniq_id, condition.type
        )
        subgraph.add_node(condition.uniq_id, node_fn)
        flush_print(f"{condition.name} {condition.uniq_id}'s condition", status=True)
        flush_print(f"true will go {condition.true_next}", status=True)
        flush_print(f"false will go {condition.false_next}", status=True)
        subgraph.add_conditional_edges(
            condition.uniq_id,
            conditional_edge,
            {
                "True": condition.true_next if condition.true_next else END,
                "False": condition.false_next if condition.false_next else END
            }
        )
    return subgraph.compile()

class MainGraphState(TypedDict):
    input: Union[str, None]

def invoke_root(state: MainGraphState):
    subgraph = subgraph_registry["root"]
    response = subgraph.invoke(
        PipelineState(
            history="",
            task="",
            condition=False
        )
    )
    return {"input": None}

def run_workflow_as_server(llm):
    with open("graph.json", 'r') as file:
        graphs = json.load(file)
    for graph in graphs:
        subgraph_name = graph.get("name")
        node_map = parse_nodes_from_json(graph)

        # Register the tool functions dynamically if has tool node, must before build graph
        for tool_node in find_nodes_by_type(node_map, "TOOL"):
            tool_code = f"{tool_node.description}"
            exec(tool_code, globals())
        subgraph = build_subgraph(node_map, llm, subgraph_name)
        subgraph_registry[subgraph_name] = subgraph

    main_graph = StateGraph(MainGraphState)
    main_graph.add_node("subgraph", invoke_root)
    main_graph.set_entry_point("subgraph")
    main_graph = main_graph.compile()

    for state in main_graph.stream({"input": None}):
        flush_print(str(state), status=False)
