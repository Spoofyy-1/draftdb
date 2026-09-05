from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import json, os, math
HERE=os.path.dirname(os.path.abspath(__file__)); app=FastAPI()
@app.get("/api/runs")
def runs():
    p=f"{HERE}/lab_runs.jsonl"
    if not os.path.exists(p): return JSONResponse([])
    def clean(o):
        if isinstance(o,float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    out=[]
    for line in open(p):
        line=line.strip()
        if not line: continue
        try: out.append(clean(json.loads(line)))   # NaN/Inf -> null so strict JSON serialises
        except Exception: pass                      # skip partially-written lines
    return JSONResponse(out)
@app.get("/",response_class=HTMLResponse)
def index():
    return open(f"{HERE}/dash.html").read()

@app.get("/api/stack")
def stack():
    out={"styles":{}}
    for fn in ("best_results.json","r7/overnight_results.json","r7/r7_results.json"):
        p=f"{HERE}/{fn}"
        if os.path.exists(p):
            try: out["styles"].update(json.load(open(p)).get("styles",{}))
            except Exception: pass
    return JSONResponse(out)
@app.get("/stack",response_class=HTMLResponse)
def stackpage(): return open(f"{HERE}/stack.html").read()

@app.get("/api/ledger")
def ledger():
    p=f"{HERE}/vault/ledger.jsonl"; n=sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
    return JSONResponse({"blind_evaluations":n})
