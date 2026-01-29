import os
import uuid
import json
import numpy as np

DATA_DIR = os.environ.get("SFM_SERVICE_DATA", "./_runs")


def new_job():
    job_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return job_id, path


def save_json(path, name, obj):
    tmp_path = os.path.join(path, f".{name}.tmp")
    final_path = os.path.join(path, name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def load_json(path, name):
    with open(os.path.join(path, name), "r", encoding="utf-8") as f:
        return json.load(f)


def update_status(path, status: str, progress: int, detail: str = None):
    payload = {"status": status, "progress": int(progress)}
    if detail:
        payload["detail"] = detail
    save_json(path, "status.json", payload)


def load_status(path):
    status_path = os.path.join(path, "status.json")
    if not os.path.exists(status_path):
        return {"status": "idle", "progress": 0}
    try:
        return load_json(path, "status.json")
    except json.JSONDecodeError:
        return {"status": "running", "progress": 0, "detail": "starting"}


def save_npy(path, name, arr):
    np.save(os.path.join(path, name), arr)


def load_npy(path, name):
    return np.load(os.path.join(path, name), allow_pickle=False)
