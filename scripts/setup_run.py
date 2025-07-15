import os
import sys
import shutil
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "train_config")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def setup_run(config_name):
    base_name = os.path.splitext(config_name)[0]
    config_path = os.path.join(CONFIG_DIR, config_name)
    assert os.path.exists(config_path), f"Config not found: {config_path}"

    config_output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(config_output_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(config_output_dir) if d.startswith("run")]
    next_id = 1 if not existing_runs else max(int(d[3:]) for d in existing_runs) + 1
    run_id = f"run{next_id}"
    run_dir = os.path.join(config_output_dir, run_id)

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "final"), exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["run_id"] = run_id

    run_config_path = os.path.join(run_dir, "train_config.yaml")
    with open(run_config_path, "w") as f:
        yaml.dump(config, f)

    print(f"✅ Created run directory: {run_dir}")
    return run_dir

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/setup_run.py shakespeare_1m.yaml")
        sys.exit(1)
    setup_run(sys.argv[1])