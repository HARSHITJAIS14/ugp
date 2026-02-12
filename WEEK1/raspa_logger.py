import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

class RaspaLogger:
    def __init__(self, log_path: str = None):
        if log_path is None:
            # Use absolute path to script directory
            self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raspa_log.json")
        else:
            self.log_path = log_path
        self._ensure_log_file()

    def _ensure_log_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({"history": []}, f, indent=2)

    def log_request(self, request_data: Dict[str, Any]) -> int:
        log = self._read_log()
        request_id = len(log["history"]) + 1
        entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "query": request_data.get("query"),
            "steps": [],
            "final_result": None
        }
        log["history"].append(entry)
        self._write_log(log)
        return request_id

    def log_step(self, request_id: int, step_name: str, details: Dict[str, Any], time_taken: float):
        log = self._read_log()
        for entry in log["history"]:
            if entry["request_id"] == request_id:
                entry["steps"].append({
                    "step": step_name,
                    "details": details,
                    "time_taken": time_taken,
                    "timestamp": datetime.now().isoformat()
                })
                break
        self._write_log(log)

    def log_final_result(self, request_id: int, result: Any):
        log = self._read_log()
        for entry in log["history"]:
            if entry["request_id"] == request_id:
                entry["final_result"] = result
                entry["completed_at"] = datetime.now().isoformat()
                break
        self._write_log(log)

    def _read_log(self) -> Dict[str, Any]:
        with open(self.log_path, "r") as f:
            return json.load(f)

    def _write_log(self, log: Dict[str, Any]):
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=2)

# Example usage:
# logger = RaspaLogger()
# req_id = logger.log_request({"query": "Run simulation X"})
# start = time.time()
# ... do something ...
# logger.log_step(req_id, "llm_called", {"prompt": "..."}, time.time() - start)
# logger.log_final_result(req_id, {"output": "Simulation complete"})
