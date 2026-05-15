import json
import os
import random
from Utils.memory_utils import load_memory
from Utils.memory_utils import save_memory
from Utils.memory_utils import update_memory
from Utils.memory_utils import get_hardware_profile

MEMORY_FILE = "kernel_memory.json"

global_strategy_memory = {
    "tiling": [],
    "unrolling": [],
    "memory": []
}

def rule_based_strategy(features):
    
    if features["num_loops"] >= 2:
        return "unrolling"   
    if features["has_strided_access"]:
        return "memory"
    
    return "unrolling"

def select_strategy(features, iteration, max_iter):
    mem_strategy = retrieve_best_strategy(features)
    if mem_strategy:
        return mem_strategy
    # Exploration → Exploitation
    explore_prob = max(0.1, 1 - (iteration / max_iter))

    if random.random() < explore_prob:
        return random.choice(["tiling", "unrolling", "memory"])

    # Exploitation (best historical)
    avg_perf = {}

    for s, values in global_strategy_memory.items():
        if values:
            times = [v["time"] for v in values if "time" in v]
    
            if times:
                avg_perf[s] = sum(times) / len(times)
            else:
                avg_perf[s] = float("inf")
    
        else:
            avg_perf[s] = float("inf")

    return min(avg_perf, key=avg_perf.get)

def retrieve_best_strategy(features):
    memory = load_memory()

    best_score = float("inf")
    best = None
    
    for m in memory:

            if m.get("kernel_type") != features.get("kernel_type"):
                continue
            current_pattern = "strided" if features["has_strided_access"] else "coalesced"
        
            if m.get("memory_pattern") != current_pattern:
                continue
        
            score_penalty = 0
        
            if abs(m.get("num_loops", 0) - features["num_loops"]) > 1:
                score_penalty += 0.2
        
            if m.get("kernel_dim") != features.get("kernel_dim"):
                score_penalty += 0.2
        
            if m.get("uses_shared_memory") != features["uses_shared_memory"]:
                score_penalty += 0.1

            if "mep_time" in m:
                if m["mep_time"] is None or m["mep_time"] <= 0:
                    continue
                score = m["mep_time"] * (1 + score_penalty)
        
            elif "real_speedup" in m:
                if m["real_speedup"] is None or m["real_speedup"] <= 0:
                    continue
                score = (1.0 / m["real_speedup"]) * (1 + score_penalty)
        
            else:
                continue
        
            if score < best_score:
                best_score = score
                best = m["optimization"]
    
    return best    

def get_best_strategy_from_history(features):
    best_strategy = None
    best_time = float("inf")

    for strategy, entries in global_strategy_memory.items():
        for entry in entries:
            f = entry["features"]

            if (
                f["num_loops"] == features["num_loops"] and
                f["is_2d"] == features["is_2d"]
            ):
                if entry["time"] < best_time:
                    best_time = entry["time"]
                    best_strategy = strategy

    return best_strategy

