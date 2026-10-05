# main.py
"""
Entry point for the CatFunc pipeline.
Reads config.py and runs selected stages in order.
"""

import os
import config

from stages import analyze, expand, validate, predict, survey

# -----------------------------------------------------------------------------
# Build full file paths
# -----------------------------------------------------------------------------
def resolve_paths(cfg):
    paths = {}
    for key, filename in cfg.INPUT_FILES.items():
        paths[key] = os.path.join(cfg.DATA_DIR, filename)
    paths["embeddings"]      = os.path.join(cfg.METADATA_DIR, "embeddings")
    paths["analyze_dir"]     = os.path.join(cfg.ANALYSIS_DIR, "analyze")
    paths["updated_literature"] = os.path.join(cfg.ANALYSIS_DIR, "analyze", "updated_literature.csv")
    paths["expand_dir"]      = os.path.join(cfg.ANALYSIS_DIR, "expand")
    paths["expand_queries"]  = os.path.join(cfg.METADATA_DIR, "expand", "queries")
    paths["expand_hmm_dir"]  = os.path.join(cfg.METADATA_DIR, "expand")
    paths["expand_combine_dir"] = os.path.join(cfg.METADATA_DIR, "expand", "combine")
    paths["validate_dir"]    = os.path.join(cfg.ANALYSIS_DIR, "validate")
    paths["predict_dir"]     = os.path.join(cfg.ANALYSIS_DIR, "predict")
    paths["survey_hits_dir"] = os.path.join(cfg.METADATA_DIR, "survey", "hits")
    paths["survey_dir"]      = os.path.join(cfg.ANALYSIS_DIR, "survey")
    return paths

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    paths  = resolve_paths(config)
    stages = config.STAGES
    params = config.PARAMS
    hpc    = config.HPC

    # create core directories
    os.makedirs(config.DATA_DIR,     exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)

    if stages.get("analyze", {}).get("run", False):
        analyze.run(paths, params.get("analyze", {}), hpc)

    if stages.get("expand", {}).get("run", False):
        substages = stages["expand"].get("substages", {})
        expand.run(paths, params.get("expand", {}), hpc, substages=substages)

    if stages.get("validate", {}).get("run", False):
        substages = stages["validate"].get("substages", {})
        validate.run(paths, params.get("validate", {}), hpc, substages=substages)

    if stages.get("predict", {}).get("run", False):
        predict.run(paths, params.get("predict", {}), hpc)

    if stages.get("survey", {}).get("run", False):
        substages = stages["survey"].get("substages", {})
        survey.run(paths, params.get("survey", {}), hpc, substages=substages)

if __name__ == "__main__":
    main()
