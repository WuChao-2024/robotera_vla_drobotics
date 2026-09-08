#!/usr/bin/env python3
"""droboticflow POC platform bootstrap for the robotera_vla operators.

Subcommands:
  register          register norm_stats + train operators from
                    operator-definitions/*.json (IMAGE_PLACEHOLDER replaced)
  workflow          create project + two-node workflow (norm_stats -> train)
                    with artifact storage policy
  run               trigger a run of the workflow (optional --overrides JSON)
  status            list robotera operators / workflows

Usage:
  python3 platform_bootstrap.py register --image <full-tag>
  python3 platform_bootstrap.py workflow
  python3 platform_bootstrap.py run [--node-overrides '{"train-node": {...}}']
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.request
import uuid

BASE = "http://120.48.164.19"
ACCOUNTS = {
    "pipeline": ("poc-acceptance-pipeline@droboticflow.local", "youbixuanpoc"),
    "admin": ("poc-acceptance-admin@droboticflow.local", "youbixuanpoc"),
}
EMAIL, PASSWORD = ACCOUNTS[os.environ.get("OP_ACCOUNT", "pipeline")]
HERE = pathlib.Path(__file__).resolve().parent
DEFS = HERE.parent / "operator-definitions"
COOKIE_JAR = pathlib.Path("/tmp/rv_poc_cookies.txt")

PROJECT_NAME = "Robotera M7 pi0.5 训练验收"
WORKFLOW_NAME = "M7 pi0.5 norm_stats -> train"
DATASET_DEFAULT = "/workspace/cauchy_dev/robotera_opensource/M7_pickplace_example"
STORAGE_CONNECTION_ID = "storage-485ec27477c334ada4372c3fca91154a"
STORAGE_BUCKET = "testxyz"


class Client:
    def __init__(self):
        self.cookies = {}
        self.csrf = ""

    def login(self):
        body = json.dumps({"email": EMAIL, "password": PASSWORD}).encode()
        req = urllib.request.Request(
            f"{BASE}/api/auth/login", data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=60)
        self.cookies = {
            c.name: c.value for c in resp.cookies
        } if hasattr(resp, "cookies") else dict(
            item.split("=", 1) for item in
            (resp.headers.get("Set-Cookie") or "").split("; ")
            if "=" in item
        )
        # fall back: parse set-cookie lines properly
        self.cookies = {}
        for header in resp.headers.get_all("Set-Cookie") or []:
            pair = header.split(";", 1)[0]
            if "=" in pair:
                k, v = pair.split("=", 1)
                self.cookies[k.strip()] = v.strip()
        self.csrf = self.cookies.get("droboticflow_csrf", "")
        if not self.csrf:
            raise RuntimeError("login succeeded but no CSRF cookie")
        COOKIE_JAR.write_text(json.dumps(self.cookies))

    def req(self, method, path, payload=None):
        headers = {
            "X-Droboticflow-CSRF-Token": self.csrf,
            "Cookie": "; ".join(f"{k}={v}" for k, v in self.cookies.items()),
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                text = resp.read().decode()
                return json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise RuntimeError(f"{method} {path} -> {exc.code}: {body[:2000]}") from exc

    def get(self, path):
        return self.req("GET", path)

    def post(self, path, payload):
        return self.req("POST", path, payload)

    def put(self, path, payload):
        return self.req("PUT", path, payload)


def list_operators(client):
    data = client.get("/api/operators")
    if isinstance(data, dict):
        return data.get("items") or data.get("operators") or []
    return data if isinstance(data, list) else []


def cmd_register(args, client):
    image = args.image
    created = []
    for name in ("norm_stats_operator.json", "train_operator.json"):
        body = json.loads((DEFS / name).read_text())
        body["image"] = image
        existing = [o for o in list_operators(client) if o.get("name") == body["name"]]
        if existing:
            op_id = existing[0]["id"]
            if not getattr(args, "update", False):
                print(f"[register] '{body['name']}' already exists: {op_id} (skip; use --update to refresh)")
                created.append(existing[0])
                continue
            out = client.put(f"/api/operators/{op_id}", body)
            print(f"[register] updated '{body['name']}': {op_id} -> image {image}")
            created.append(out or {"name": body["name"], "id": op_id})
            continue
        out = client.post("/api/operators", body)
        print(f"[register] created '{body['name']}': {out.get('id')}")
        created.append(out)
    pathlib.Path("/tmp/rv_operator_ids.json").write_text(
        json.dumps({c["name"]: c["id"] for c in created})
    )
    return created


def list_projects(client):
    data = client.get("/api/projects")
    if isinstance(data, dict):
        return data.get("items") or data.get("projects") or []
    return data if isinstance(data, list) else []


def ensure_project(client):
    for p in list_projects(client):
        if p.get("name") == PROJECT_NAME:
            print(f"[workflow] reuse project {p['id']}")
            return p["id"]
    out = client.post("/api/projects", {"name": PROJECT_NAME})
    print(f"[workflow] created project {out['id']}")
    return out["id"]


def dataset_port_ids(count):
    return ["dataset"] + [f"dataset_{i}" for i in range(2, count + 1)]


def scan_node_from_operator(scan_op, index, source_path):
    return {
        "id": f"dir-scan-{index}",
        "type": "operator",
        "data": {
            "label": f"目录读取-{index}",
            "category": scan_op.get("category", "IO"),
            "description": scan_op.get("description", ""),
            "inputs": scan_op.get("inputs", []),
            "outputs": scan_op.get("outputs", []),
            "operatorId": scan_op["id"],
            "operatorVersion": scan_op.get("version", "v1"),
            "image": scan_op.get("image", ""),
            "containerCommand": scan_op.get("containerCommand"),
            "hyperparams": [
                {"key": "SOURCE_PATH", "type": "string", "value": source_path,
                 "description": "数据集目录的 BOS 引用"},
            ],
            "connectionId": STORAGE_CONNECTION_ID,
            "bucket": STORAGE_BUCKET,
        },
        "position": {"x": -500, "y": index * 180},
    }


# 平台 2026-09 中将注册表里的目录扫描算子改为仅 json 概览(无 directory 输出)后,
# 依据"工作流节点冻结算子快照"的官方机制, 复刻旧快照继续提供 directory 输出。
LEGACY_SCAN_SNAPSHOT = {
    "category": "IO",
    "description": "读取必填目录输入；支持 BOS directory ref 与本地目录，输出完整文件清单供下游算子消费，不保存增量扫描状态。",
    "inputs": [
        {"id": "sourceDirectory", "name": "sourceDirectory", "type": "file", "required": True,
         "description": "真实目录输入（BOS directory ref）", "transmission": "oss"},
    ],
    "outputs": [
        {"id": "directory", "name": "directory", "type": "file", "required": False,
         "description": "目录引用；技术 manifest 仅内部索引", "transmission": "oss"},
        {"id": "videos", "name": "videos", "type": "video", "required": False,
         "description": "目录内视频集合", "transmission": "oss"},
        {"id": "images", "name": "images", "type": "image", "required": False,
         "description": "目录内图片集合", "transmission": "oss"},
        {"id": "files", "name": "files", "type": "file", "required": False,
         "description": "目录内其他文件集合", "transmission": "oss"},
        {"id": "summary", "name": "summary", "type": "json", "required": False,
         "description": "目录读取摘要", "transmission": "bus"},
    ],
    "operatorId": "operator-builtin-directory-scan",
    "operatorVersion": "v1",
    "image": "ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/droboticflow-operator-directory-scan:5534416-poc-fanyu-v54-r3",
    "containerCommand": ["python", "/app/backend/operator-runner/main.py"],
}


def legacy_scan_node(index, source_path):
    data = {
        "label": f"目录读取-{index}",
        "hyperparams": [
            {"key": "SOURCE_PATH", "type": "string", "value": source_path,
             "description": "数据集目录的 BOS 引用"},
        ],
        "connectionId": STORAGE_CONNECTION_ID,
        "bucket": STORAGE_BUCKET,
    }
    data.update(LEGACY_SCAN_SNAPSHOT)
    return {
        "id": f"dir-scan-{index}",
        "type": "operator",
        "data": data,
        "position": {"x": -500, "y": index * 180},
    }


def cmd_workflow(args, client):
    ops = {o["name"]: o for o in list_operators(client)}
    norm_op = ops.get("M7 pi0.5 归一化统计")
    train_op = ops.get("M7 pi0.5 训练")
    scan_op = None
    for o in list_operators(client):
        if o.get("id") == "operator-builtin-directory-scan":
            scan_op = o
            break
    if not norm_op or not train_op:
        raise RuntimeError("operators not registered; run `register` first")

    dataset_paths = [p.strip() for p in args.dataset_paths.split(",") if p.strip()]

    # 目录读取算子仍有 directory 输出时用注册表定义; 平台将其改为仅 json 概览时,
    # 用冻结快照复刻的 legacy 节点(平台按节点快照执行, 旧工作流即此机制)
    scan_has_directory = scan_op is not None and any(
        p.get("id") == "directory" for p in (scan_op.get("outputs") or [])
    )

    # v1.1.0 端口契约: 算子只声明 assets(第一,必须) + dataset(第二,必须);
    # 第 3..N 个数据集端口在此合成, id 为随机 UUID —— 与用户在 UI 上手工
    # "添加输入端口"产生的端口完全同构(适配器不看端口名, 均视为数据集)
    def dataset_ports_for(op, count):
        declared = [p for p in op.get("inputs", []) if p.get("id") == "dataset"]
        ports = [dict(p) for p in declared[:count]]
        while len(ports) < count:
            ports.append({
                "id": str(uuid.uuid4()),
                "name": f"数据集 {len(ports) + 1}",
                "type": "file",
                "container": "directory",
                "payloadMode": "ref",
                "required": False,
                "description": "用户自增数据集端口(等效 UI 手工添加; 适配器不识别端口名, 任意已连线端口均视为数据集)",
            })
        return ports[:count]

    project_id = ensure_project(client)

    nodes = []
    edges = []
    workflow_inputs = []
    norm_ds_ports = dataset_ports_for(norm_op, max(len(dataset_paths), 1))
    train_ds_ports = dataset_ports_for(train_op, max(len(dataset_paths), 1))
    if dataset_paths:
        make_scan = (lambda i, path: scan_node_from_operator(scan_op, i, path)) if scan_has_directory else legacy_scan_node
        for i, path in enumerate(dataset_paths, start=1):
            nodes.append(make_scan(i, path))
            edges.append({"id": f"e-scan{i}-norm", "source": f"dir-scan-{i}",
                          "sourceHandle": "directory", "target": "norm-stats-node",
                          "targetHandle": norm_ds_ports[i - 1]["id"]})
            edges.append({"id": f"e-scan{i}-train", "source": f"dir-scan-{i}",
                          "sourceHandle": "directory", "target": "train-node",
                          "targetHandle": train_ds_ports[i - 1]["id"]})
            # 工作流输入: BOS 目录 ref 默认值, 喂给目录读取的必填 sourceDirectory
            bos = path[len("bos://"):] if path.startswith("bos://") else f"{STORAGE_BUCKET}/{path.strip('/')}"
            bucket, _, prefix = bos.partition("/")
            workflow_inputs.append({
                "id": f"ds-{i}",
                "name": f"数据集目录 {i}",
                "description": f"子数据集 {prefix.rstrip('/').split('/')[-1]} 的 BOS 目录引用",
                "type": "file",
                "container": "directory",
                "payloadMode": "ref",
                "required": True,
                "defaultPayload": {
                    "mode": "ref",
                    "type": "file",
                    "ref": {
                        "kind": "directory",
                        "provider": "BOS",
                        "connectionId": STORAGE_CONNECTION_ID,
                        "bucket": bucket,
                        "prefix": prefix.strip("/"),
                    },
                },
                "bindings": [{"nodeId": f"dir-scan-{i}", "targetPortId": "sourceDirectory"}],
            })

    norm_node = {
        "id": "norm-stats-node",
        "type": "operator",
        "data": {
            "label": "M7 归一化统计",
            "category": norm_op.get("category", "AI"),
            "description": norm_op.get("description", ""),
            "inputs": norm_ds_ports,
            "outputs": norm_op.get("outputs", []),
            "operatorId": norm_op["id"],
            "operatorVersion": norm_op.get("version", "1.1.0"),
            "image": norm_op.get("image", ""),
            "containerCommand": norm_op.get("containerCommand"),
            "hyperparams": norm_op.get("hyperparams", []),
            "cpu": "16", "memory": "48Gi", "gpu": "",
        },
        "position": {"x": 0, "y": 0},
    }
    train_gpu = os.environ.get("WORKFLOW_TRAIN_GPU", "8")
    train_mem = os.environ.get("WORKFLOW_TRAIN_MEM", "512Gi")
    # assets 端口排第一(契约: 第一输入归一化统计), 数据集端口随其后
    assets_port = [p for p in train_op.get("inputs", []) if p.get("id") == "assets"]
    train_node = {
        "id": "train-node",
        "type": "operator",
        "data": {
            "label": "M7 pi0.5 训练",
            "category": train_op.get("category", "AI"),
            "description": train_op.get("description", ""),
            "inputs": assets_port + train_ds_ports,
            "outputs": train_op.get("outputs", []),
            "operatorId": train_op["id"],
            "operatorVersion": train_op.get("version", "1.1.0"),
            "image": train_op.get("image", ""),
            "containerCommand": train_op.get("containerCommand"),
            "hyperparams": train_op.get("hyperparams", []),
            "cpu": "32", "memory": train_mem, "gpu": train_gpu,
        },
        "position": {"x": 400, "y": 0},
    }
    nodes += [norm_node, train_node]
    edges.append({
        "id": "edge-assets",
        "source": "norm-stats-node",
        "sourceHandle": "assets",
        "target": "train-node",
        "targetHandle": "assets",
    })
    payload = {
        "projectId": project_id,
        "name": getattr(args, "name", "") or WORKFLOW_NAME,
        "description": "目录读取(子数据集) -> 归一化统计 -> pi0.5 训练",
        "type": "batch",
        "nodes": nodes,
        "edges": edges,
        "workflowInputs": workflow_inputs,
        "nodeTimeoutSeconds": args.node_timeout,
        "runTimeoutSeconds": args.run_timeout,
        "artifactStorage": {
            "connectionId": STORAGE_CONNECTION_ID,
            "bucket": STORAGE_BUCKET,
            "prefixTemplate": "robotera-vla/{runId}",
        },
    }
    wf = client.post(f"/api/projects/{project_id}/workflows", payload)
    print(f"[workflow] created workflow {wf.get('id')} with {len(nodes)} nodes / {len(edges)} edges")
    pathlib.Path("/tmp/rv_workflow_id.txt").write_text(wf.get("id", ""))
    return wf


def cmd_run(args, client):
    wf_id = args.workflow or pathlib.Path("/tmp/rv_workflow_id.txt").read_text().strip()
    payload = {}
    if args.node_overrides:
        payload["nodeOverrides"] = json.loads(args.node_overrides)
    if args.debug_node:
        payload["mode"] = "debug"
        payload["debugTarget"] = {"targetNodeId": args.debug_node, "singleNode": True}
    out = client.post(f"/api/workflows/{wf_id}/run", payload)
    print(json.dumps(out, ensure_ascii=False)[:2000])
    return out


def cmd_status(_args, client):
    ops = [o for o in list_operators(client) if "M7" in o.get("name", "")]
    for o in ops:
        print(f"operator: {o['id']} | {o['name']} | {o.get('image', '')[:80]}")
    projects = client.get("/api/projects")
    for p in projects:
        if p.get("name") == PROJECT_NAME:
            wfs = client.get(f"/api/projects/{p['id']}/workflows")
            items = wfs if isinstance(wfs, list) else wfs.get("items", [])
            for w in items:
                print(f"workflow: {w['id']} | {w.get('name')}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register"); p.add_argument("--image", required=True)
    p.add_argument("--update", action="store_true", help="refresh existing operators via PUT instead of skipping")
    p = sub.add_parser("workflow")
    p.add_argument("--name", default="", help="workflow name (default: %s)" % WORKFLOW_NAME)
    p.add_argument("--node-timeout", type=int, default=0)
    p.add_argument("--run-timeout", type=int, default=0)
    p.add_argument("--dataset-paths", default="", help="comma-separated LeRobot collection dirs; each becomes one directory-scan node")
    p = sub.add_parser("run")
    p.add_argument("--workflow", default="")
    p.add_argument("--node-overrides", default="")
    p.add_argument("--debug-node", default="")
    sub.add_parser("status")
    args = parser.parse_args()

    client = Client()
    client.login()
    {"register": cmd_register, "workflow": cmd_workflow,
     "run": cmd_run, "status": cmd_status}[args.cmd](args, client)


if __name__ == "__main__":
    main()
