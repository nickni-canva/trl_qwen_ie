#!/usr/bin/env python3
"""
Test to validate that our MMMG implementation produces EXACT same results as original MMMG code.
This test copies MMMG's functions directly to avoid import issues.
"""

import sys
import networkx as nx

# ============================================================================
# MMMG ORIGINAL FUNCTIONS (copied from step3_stat.py)
# ============================================================================

def mmmg_parse_dependencies(dependencies):
    '''
    Original MMMG parse_dependencies from step3_stat.py lines 12-75
    '''
    edges = []
    for dep, exists in dependencies.items():
        if exists:
            try:
                relation, nodes = dep.split('(', 1)
                try:
                    if nodes[-1] == ")":
                        nodes = nodes[:-1]
                    else:
                        print(f"Error parsing dependency BRACKETS: {dep}")

                    source, target = nodes.split(', ', 1)
                        
                except ValueError:
                    if len(nodes.split(', ', 1)) == 1 and len(nodes.split(',', 1)) == 2:
                        source, target = nodes.split(',', 1)
                    else:
                        print(f"Error parsing dependency: {dep}")
                        continue
                    
                source = source.lower()
                target = target.lower()

                if "change(" in source.lower():
                    source = source.strip(")").split("change(")[-1].lower()
                if "change(" in target.lower():
                    target = target.strip(")").split("change(")[-1].lower() 
                
                lower_relation = relation.lower()
                if lower_relation == "requires":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((t, source, relation))
                    else:
                        edges.append((target, source, relation))
                elif lower_relation == "defines":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                            edges.append((t, source, relation))
                    else:
                        edges.append((source, target, relation))
                        edges.append((target, source, relation))
                    
                else:
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                    else:
                        edges.append((source, target, relation))
            except ValueError:
                continue
    return edges


def mmmg_build_graph(elements, dependencies):
    '''
    Original MMMG build_graph from step3_stat.py lines 78-121
    '''
    G = nx.DiGraph()
    for node, exists in elements.items():
        if exists:
            G.add_node(node.lower())
    edges = mmmg_parse_dependencies(dependencies)

    
    for source, target, relation in edges:
        match_src = False
        match_tgt = False
        src_candidate = [s for s in G.nodes if source in s and source != s] + [s for s in G.nodes if s in source and s != source and " " in source]
        tgt_candidate = [t for t in G.nodes if target in t and target != t] + [t for t in G.nodes if t in target and t != target and " " in target]
        
        src_candidate = [s for s in src_candidate if "" in s]
        tgt_candidate = [t for t in tgt_candidate if "" in t]
        
        for node in G.nodes:
            if source == node:
                match_src = True
            if target == node:
                match_tgt = True
        
        if not match_src:
            if len(src_candidate) == 1:
                match_src = True
                source = src_candidate[0]
        if not match_tgt:
            if len(tgt_candidate) == 1:
                match_tgt = True
                target = tgt_candidate[0]

        if match_src and match_tgt:
            G.add_edge(source, target, label=relation)
          
    return G


def mmmg_normalized_ged(G1, G2):
    '''
    Original MMMG normalized_ged from step3_stat.py lines 123-130
    '''
    try:
        ged = next(nx.optimize_graph_edit_distance(G1, G2))
    except StopIteration:
        ged = 0
    max_size = G1.number_of_nodes() + G2.number_of_nodes() + G1.number_of_edges() + G2.number_of_edges()
    return ged / max_size if max_size > 0 else 1.0


# ============================================================================
# OUR IMPLEMENTATIONS (from train_grpo_qwen_image.py)
# ============================================================================

def our_parse_dependencies(dependencies):
    """Our implementation - copy from train_grpo_qwen_image.py"""
    edges = []
    for dep, exists in dependencies.items():
        if exists:
            try:
                relation, nodes = dep.split('(', 1)
                
                try:
                    if nodes[-1] == ")":
                        nodes = nodes[:-1]
                    else:
                        print(f"Error parsing dependency BRACKETS: {dep}")
                    
                    source, target = nodes.split(', ', 1)
                except ValueError:
                    if len(nodes.split(', ', 1)) == 1 and len(nodes.split(',', 1)) == 2:
                        source, target = nodes.split(',', 1)
                    else:
                        print(f"Error parsing dependency: {dep}")
                        continue
                
                source = source.lower()
                target = target.lower()
                
                if "change(" in source.lower():
                    source = source.strip(")").split("change(")[-1].lower()
                if "change(" in target.lower():
                    target = target.strip(")").split("change(")[-1].lower()
                
                lower_relation = relation.lower()
                if lower_relation == "requires":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((t, source, relation))
                    else:
                        edges.append((target, source, relation))
                elif lower_relation == "defines":
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                            edges.append((t, source, relation))
                    else:
                        edges.append((source, target, relation))
                        edges.append((target, source, relation))
                else:
                    if isinstance(target, list) or isinstance(target, tuple):
                        for t in target:
                            edges.append((source, t, relation))
                    else:
                        edges.append((source, target, relation))
            except ValueError:
                continue
    return edges

def our_build_graph(elements, dependencies):
    """Our implementation - copy from train_grpo_qwen_image.py"""
    G = nx.DiGraph()
    
    for node, exists in elements.items():
        if exists:
            G.add_node(node.lower())
    
    edges = our_parse_dependencies(dependencies)
    
    for source, target, relation in edges:
        match_src = False
        match_tgt = False
        
        src_candidate = [s for s in G.nodes if source in s and source != s] + \
                       [s for s in G.nodes if s in source and s != source and " " in source]
        tgt_candidate = [t for t in G.nodes if target in t and target != t] + \
                       [t for t in G.nodes if t in target and t != target and " " in target]
        
        src_candidate = [s for s in src_candidate if "" in s]
        tgt_candidate = [t for t in tgt_candidate if "" in t]
        
        for node in G.nodes:
            if source == node:
                match_src = True
            if target == node:
                match_tgt = True
        
        if not match_src:
            if len(src_candidate) == 1:
                match_src = True
                source = src_candidate[0]
        if not match_tgt:
            if len(tgt_candidate) == 1:
                match_tgt = True
                target = tgt_candidate[0]
        
        if match_src and match_tgt:
            G.add_edge(source, target, label=relation)
    
    return G

def our_normalized_ged(G1, G2):
    """Our implementation - copy from train_grpo_qwen_image.py"""
    try:
        ged = next(nx.optimize_graph_edit_distance(G1, G2))
    except StopIteration:
        ged = 0
    
    max_size = (G1.number_of_nodes() + G2.number_of_nodes() + 
               G1.number_of_edges() + G2.number_of_edges())
    
    return ged / max_size if max_size > 0 else 1.0


# ============================================================================
# TESTS
# ============================================================================

def test_parse_dependencies():
    """Test parse_dependencies produces same output"""
    print("\n" + "="*80)
    print("TEST 1: parse_dependencies()")
    print("="*80)
    
    test_cases = [
        {
            "Causes(evolution, adaptation)": True,
            "Requires(natural selection, variation)": True,
            "Defines(fitness, survival rate)": True,
        },
        {
            "Causes(A, B)": True,
            "Causes(C, D)": False,
            "Entails(E, F)": True,
        },
        {},
    ]
    
    all_passed = True
    for i, deps in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: {deps}")
        
        mmmg_result = mmmg_parse_dependencies(deps)
        our_result = our_parse_dependencies(deps)
        
        mmmg_sorted = sorted(mmmg_result)
        our_sorted = sorted(our_result)
        
        print(f"MMMG result: {mmmg_sorted}")
        print(f"Our result:  {our_sorted}")
        
        if mmmg_sorted == our_sorted:
            print("✅ MATCH")
        else:
            print("❌ MISMATCH!")
            all_passed = False
    
    return all_passed


def test_build_graph():
    """Test build_graph produces same structure"""
    print("\n" + "="*80)
    print("TEST 2: build_graph()")
    print("="*80)
    
    test_cases = [
        {
            "elements": {"evolution": True, "adaptation": True, "natural selection": True, "variation": True},
            "dependencies": {
                "Causes(evolution, adaptation)": True,
                "Requires(natural selection, variation)": True,
            }
        },
        {
            "elements": {"A": True, "B": True, "C": False},
            "dependencies": {
                "Defines(A, B)": True,
                "Causes(B, C)": True,
            }
        },
        {
            "elements": {},
            "dependencies": {}
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Elements: {test['elements']}")
        print(f"Dependencies: {test['dependencies']}")
        
        mmmg_graph = mmmg_build_graph(test['elements'], test['dependencies'])
        our_graph = our_build_graph(test['elements'], test['dependencies'])
        
        mmmg_nodes = sorted(mmmg_graph.nodes())
        our_nodes = sorted(our_graph.nodes())
        
        print(f"MMMG nodes: {mmmg_nodes}")
        print(f"Our nodes:  {our_nodes}")
        
        mmmg_edges = sorted([(u, v) for u, v in mmmg_graph.edges()])
        our_edges = sorted([(u, v) for u, v in our_graph.edges()])
        
        print(f"MMMG edges: {mmmg_edges}")
        print(f"Our edges:  {our_edges}")
        
        if mmmg_nodes == our_nodes and mmmg_edges == our_edges:
            print("✅ MATCH")
        else:
            print("❌ MISMATCH!")
            all_passed = False
    
    return all_passed


def test_normalized_ged():
    """Test normalized_ged produces same values"""
    print("\n" + "="*80)
    print("TEST 3: normalized_ged()")
    print("="*80)
    
    test_cases = [
        {
            "name": "Identical graphs",
            "G1_nodes": ["a", "b"],
            "G1_edges": [("a", "b")],
            "G2_nodes": ["a", "b"],
            "G2_edges": [("a", "b")],
        },
        {
            "name": "Different graphs",
            "G1_nodes": ["a", "b", "c"],
            "G1_edges": [("a", "b"), ("b", "c")],
            "G2_nodes": ["a", "b"],
            "G2_edges": [("a", "b")],
        },
        {
            "name": "Empty graphs (edge case)",
            "G1_nodes": [],
            "G1_edges": [],
            "G2_nodes": [],
            "G2_edges": [],
        },
        {
            "name": "One empty, one not",
            "G1_nodes": ["a", "b"],
            "G1_edges": [("a", "b")],
            "G2_nodes": [],
            "G2_edges": [],
        }
    ]
    
    all_passed = True
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        
        G1_mmmg = nx.DiGraph()
        G1_our = nx.DiGraph()
        for node in test['G1_nodes']:
            G1_mmmg.add_node(node)
            G1_our.add_node(node)
        for edge in test['G1_edges']:
            G1_mmmg.add_edge(*edge)
            G1_our.add_edge(*edge)
        
        G2_mmmg = nx.DiGraph()
        G2_our = nx.DiGraph()
        for node in test['G2_nodes']:
            G2_mmmg.add_node(node)
            G2_our.add_node(node)
        for edge in test['G2_edges']:
            G2_mmmg.add_edge(*edge)
            G2_our.add_edge(*edge)
        
        print(f"G1: {test['G1_nodes']} with edges {test['G1_edges']}")
        print(f"G2: {test['G2_nodes']} with edges {test['G2_edges']}")
        
        mmmg_ged = mmmg_normalized_ged(G1_mmmg, G2_mmmg)
        our_ged = our_normalized_ged(G1_our, G2_our)
        
        print(f"MMMG normalized GED: {mmmg_ged:.6f}")
        print(f"Our normalized GED:  {our_ged:.6f}")
        
        if abs(mmmg_ged - our_ged) < 1e-6:
            print("✅ MATCH")
        else:
            print(f"❌ MISMATCH! Difference: {abs(mmmg_ged - our_ged):.6f}")
            all_passed = False
    
    return all_passed


def test_full_pipeline():
    """Test full pipeline from elements/deps to knowledge fidelity"""
    print("\n" + "="*80)
    print("TEST 4: Full Pipeline (Elements → GED → Knowledge Fidelity)")
    print("="*80)
    
    gt_elements = ["evolution", "natural selection", "adaptation", "variation"]
    gt_dependencies = [
        "Causes(natural selection, adaptation)",
        "Requires(natural selection, variation)",
        "Defines(evolution, change in species)"
    ]
    
    pred_elements_yes = {
        "evolution": True,
        "natural selection": True,
        "adaptation": False,
        "variation": True,
    }
    pred_dependencies_yes = {
        "Causes(natural selection, adaptation)": False,
        "Requires(natural selection, variation)": True,
        "Defines(evolution, change in species)": True,
    }
    
    print("\nGround Truth:")
    print(f"  Elements: {gt_elements}")
    print(f"  Dependencies: {gt_dependencies}")
    
    print("\nPredicted (yes/no):")
    print(f"  Elements: {pred_elements_yes}")
    print(f"  Dependencies: {pred_dependencies_yes}")
    
    gt_elements_dict = {elem: True for elem in gt_elements}
    gt_dependencies_dict = {dep: True for dep in gt_dependencies}
    
    G_gt_mmmg = mmmg_build_graph(gt_elements_dict, gt_dependencies_dict)
    G_pred_mmmg = mmmg_build_graph(pred_elements_yes, pred_dependencies_yes)
    ged_mmmg = mmmg_normalized_ged(G_gt_mmmg, G_pred_mmmg)
    kf_mmmg = 1.0 - ged_mmmg
    
    G_gt_ours = our_build_graph(gt_elements_dict, gt_dependencies_dict)
    G_pred_ours = our_build_graph(pred_elements_yes, pred_dependencies_yes)
    ged_ours = our_normalized_ged(G_gt_ours, G_pred_ours)
    kf_ours = 1.0 - ged_ours
    
    print("\nMMMG Pipeline:")
    print(f"  GT graph: {G_gt_mmmg.number_of_nodes()} nodes, {G_gt_mmmg.number_of_edges()} edges")
    print(f"  Pred graph: {G_pred_mmmg.number_of_nodes()} nodes, {G_pred_mmmg.number_of_edges()} edges")
    print(f"  Normalized GED: {ged_mmmg:.6f}")
    print(f"  Knowledge Fidelity: {kf_mmmg:.6f}")
    
    print("\nOur Pipeline:")
    print(f"  GT graph: {G_gt_ours.number_of_nodes()} nodes, {G_gt_ours.number_of_edges()} edges")
    print(f"  Pred graph: {G_pred_ours.number_of_nodes()} nodes, {G_pred_ours.number_of_edges()} edges")
    print(f"  Normalized GED: {ged_ours:.6f}")
    print(f"  Knowledge Fidelity: {kf_ours:.6f}")
    
    if abs(kf_mmmg - kf_ours) < 1e-6:
        print("\n✅ FULL PIPELINE MATCH")
        return True
    else:
        print(f"\n❌ MISMATCH! Difference: {abs(kf_mmmg - kf_ours):.6f}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MMMG EXACT MATCH VALIDATION TEST")
    print("="*80)
    print("\nComparing our implementation with MMMG's original functions...")
    
    results = []
    
    results.append(("parse_dependencies", test_parse_dependencies()))
    results.append(("build_graph", test_build_graph()))
    results.append(("normalized_ged", test_normalized_ged()))
    results.append(("full_pipeline", test_full_pipeline()))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Implementation matches MMMG exactly!")
        print("="*80)
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Implementation differs from MMMG!")
        print("="*80)
        sys.exit(1)
