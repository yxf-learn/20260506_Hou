import json, os, tempfile
import yaml


def read_yaml(path):
    with open(path) as f: return yaml.safe_load(f)
def write_yaml(path, obj):
    with open(path, "w") as f: yaml.safe_dump(obj, f, sort_keys=False)
def read_json(path):
    with open(path) as f: return json.load(f)
def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
def atomic_write(path, content):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d)
    with os.fdopen(fd, "w") as f: f.write(content)
    os.replace(tmp, path)
