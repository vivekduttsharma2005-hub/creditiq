import json
from datetime import datetime, timezone
from pathlib import Path


class RequestLog:
    def __init__(self, path):
        self.path = Path(path) if path else None      # None ya "" = logging band

    def write(self, applicants, results):
        if not self.path:
            return
        try:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            lines = [json.dumps({"ts": now, "input": a, "default_probability": r["default_probability"],
                                 "decision": r["decision"]}) for a, r in zip(applicants, results)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass


def read_log(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
