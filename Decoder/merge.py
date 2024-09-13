"""
This file means to merge simplex bricks in the decoded LDR file.
"""

import itertools
import time
from typing import List

import os

from Utils.ModelUtils.Connection.conntype import ConnType
from Utils.ModelUtils.Connection.cpoints_util import get_trans_matrices, modify_cpoints
from Utils.ModelUtils.IO.io_util import write_bricks_to_file
from Utils.ModelUtils.brick_factory import create_brick
from Utils.ModelUtils.brick_instance import BrickInstance
from Utils.ModelUtils.unit import Unit

from gurobipy import GRB
import gurobipy as gp
import networkx as nx

import numpy as np

brick_names_need_to_be_merged = ["3024", "3022", "3023", "3623", "3070b", "3069b", "3068b"]

def construct_reference_graph(bricks: List[BrickInstance]):
    """
    Given a set of bricks, construct a reference graph.
    """
    bricks_need_merge = separate_bricks_need_to_be_merged_and_those_are_not(bricks)[0]

    G = nx.Graph()

    # 1. Add nodes (only add atomic bricks)
    for brick in bricks_need_merge:
        modify_cpoints(brick.template)
        units = brick.get_current_units()
        assert len(units) == 1
        G.add_node(brick.id)  # Supposedly, brick.id is unique.
        G.nodes[brick.id]["unit"] = units[0]
        G.nodes[brick.id]["brick"] = brick
        G.nodes[brick.id]["floating"] = True

    # 2. Add edges: Edge only exists between bricks with virtual side connectors.
    for b_i, b_j in itertools.product(bricks_need_merge, bricks):
        b_i_id, b_j_id = b_i.id, b_j.id
        conn_points = b_i.connect(b_j)
        if conn_points is None:
            continue
        for cp in conn_points:
            if cp[0] in [
                ConnType.SIDE_3024,
                ConnType.SIDE_3070,
            ]:
                G.add_edge(b_i_id, b_j_id)
                break

            elif cp[0] == ConnType.STUD_TUBE:
                G.nodes[b_i_id]["floating"] = False

    # 3. remove "floating" nodes that are actually on the ground
    max_y = max([brick.get_brick_voxels()[:, 1].max() for brick in bricks])
    for node in G.nodes:
        brick: BrickInstance = G.nodes[node]["brick"]
        brick_voxels = brick.get_brick_voxels()
        if brick_voxels[:, 1].max() == max_y:
            G.nodes[node]["floating"] = False

    return G


def get_occupancy_set(brick: BrickInstance, reference_graph: nx.Graph):
    """
    Given a brick,
    return a tuple of integers, which are the indexes of the units in the node_units list that this brick occupies.
    """
    occupancy_tuple = []
    for unit in brick.get_current_units():
        for node in reference_graph.nodes:
            node_unit = reference_graph.nodes[node]["unit"]
            if unit == node_unit:
                occupancy_tuple.append(node)
                break
    return set(occupancy_tuple)


def construct_superset(model, reference_graph: nx.Graph):
    non_atomic_bricks = ["3024", "3022", "3023", "3623", "3070b", "3069b", "3068b"]
    superset = nx.Graph()
    superset_bricks = []

    node_units: List[Unit] = [
        reference_graph.nodes[node]["unit"] for node in reference_graph.nodes
    ]
    occupancy_sets = []

    # 1. enumerate all possible bricks
    for node_unit in node_units:
        for brick_type in non_atomic_bricks:
            template_brick = create_brick(brick_type, trans_mat=np.identity(4))
            modify_cpoints(template_brick.template)
            template_units = template_brick.get_current_units()

            # check if the unit shape is the same
            if template_units[0].shape != node_unit.shape:
                continue

            # 2. enumerate all possible transformations from the node_unit to the template_unit
            for template_unit in template_units:
                trans_list = get_trans_matrices(node_unit, template_unit)
                for mat in trans_list:
                    mat = np.round(mat).astype(np.int32)
                    new_brick = create_brick(template_brick.template.id, trans_mat=mat)

                    # Check if all units of the new_brick are in the node_units collection
                    if any(
                        [
                            new_unit not in node_units
                            for new_unit in new_brick.get_current_units()
                        ]
                    ):
                        continue

                    # get the node occupancy set in the reference_graph for the new brick
                    occupancy_set = get_occupancy_set(new_brick, reference_graph)
                    if occupancy_set in occupancy_sets:
                        continue

                    # assign the color for the new brick
                    all_unit_colors = [
                        reference_graph.nodes[n]["brick"].color for n in occupancy_set
                    ]
                    new_brick.color = all_unit_colors[0]

                    occupancy_sets.append(occupancy_set)

                    superset_bricks.append(new_brick)
                    superset.add_node(new_brick.id)
                    superset.nodes[new_brick.id]["brick"] = new_brick
                    superset.nodes[new_brick.id]["occupancy_set"] = occupancy_set
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{new_brick.id}")
                    superset.nodes[new_brick.id]["var"] = var

    return superset


def set_objective(model, superset: nx.Graph):
    """
    The only objective is to minimize the number of bricks used
    """
    variables = [superset.nodes[node]["var"] for node in superset.nodes]
    model.setObjective(gp.quicksum(variables), GRB.MINIMIZE)


def find_nearest_non_floating_node(node, reference_graph: nx.Graph):
    if not reference_graph.nodes[node]["floating"]:
        return node

    for edge in nx.bfs_edges(reference_graph, node):
        if not reference_graph.nodes[edge[1]]["floating"]:
            return edge[1]


def set_constraints(model, reference_graph: nx.Graph, superset: nx.Graph):
    len_constraints = 0
    # 1. Coverage & non-overlapping
    for node in reference_graph.nodes:
        if reference_graph.nodes[node]["floating"]:
            nearest_non_floating_node = find_nearest_non_floating_node(
                node, reference_graph
            )

            # find all the variables in the superset that cover this floating node
            # and the nearest non-floating node
            variables_covering_this_node = [
                superset.nodes[s]["var"]
                for s in superset.nodes
                if node in superset.nodes[s]["occupancy_set"]
                and nearest_non_floating_node in superset.nodes[s]["occupancy_set"]
            ]

        else:
            # find all the variables in the superset that cover this non-floating node
            variables_covering_this_node = [
                superset.nodes[s]["var"]
                for s in superset.nodes
                if node in superset.nodes[s]["occupancy_set"]
            ]

        # at last, the node must be covered by one and only one variable
        model.addConstr(
            gp.quicksum(variables_covering_this_node) == 1, name=f"cov_{node}"
        )
        len_constraints += 1

    print(f"Model has {len_constraints} constraints")


def interpret_solution(superset: nx.Graph, residual_bricks: List[BrickInstance]):
    selected_bricks = [
        superset.nodes[node]["brick"]
        for node in superset.nodes
        if superset.nodes[node]["var"].x > 0.5
    ]
    return selected_bricks + residual_bricks


def separate_bricks_need_to_be_merged_and_those_are_not(bricks: List[BrickInstance]):
    bricks_need_merge = [
        b
        for b in bricks
        if b.template.id in brick_names_need_to_be_merged
    ]
    bricks_not_need_merge = [
        b
        for b in bricks
        if b.template.id
        not in brick_names_need_to_be_merged
    ]
    return bricks_need_merge, bricks_not_need_merge


def merge_bricks(bricks: List[BrickInstance]):
    start_time = time.time()
    model = gp.Model("merge")
    print("constructing the reference graph")
    G = construct_reference_graph(bricks)

    print("constructing the superset")
    superset = construct_superset(model, G)

    print("setting the objective")
    set_objective(model, superset)

    print("setting the constraints")
    set_constraints(model, G, superset)
    model.optimize()

    residual_bricks = separate_bricks_need_to_be_merged_and_those_are_not(bricks)[1]
    merged_bricks = interpret_solution(superset, residual_bricks)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Merge time: {duration}")

    return merged_bricks, duration
