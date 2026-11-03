"""Offline unit and integration checks; forecasts are always mocked."""
import copy
import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import patch
from urllib.error import URLError
from .carbon_intensity import parse_forecast, fetch_forecast, period_start, ForecastError
from .estimator import operational, sci, compare_periods, estimate
from .ecologits_adapter import estimate_energy

NOW = datetime(2026, 10, 29, 10, 14, tzinfo=timezone.utc)


def payload(now=NOW, values=(200,150,100,100,180,220)):
    start = period_start(now)
    return {"data":[{"from":(start+timedelta(minutes=30*i)).isoformat(), "to":(start+timedelta(minutes=30*(i+1))).isoformat(), "intensity":{"forecast":v,"actual":999}} for i,v in enumerate(values)]}


class CarbonTests(unittest.TestCase):
    def test_units(self):
        self.assertAlmostEqual(operational(2/1000,200),.4)
        self.assertAlmostEqual(sci(.4,.1),.5)
        with self.assertRaises(ValueError): sci(0,0,0)

    def test_invalid_numbers(self):
        for x in (-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError): operational(x,100)

    def test_recommendation(self):
        result=compare_periods(.001,.1,parse_forecast(payload(),NOW))
        rec=result["recommendation"]
        self.assertEqual(rec["start"],result["periods"][2]["start"])
        self.assertAlmostEqual(rec["reductionGCO2eApprox"],.1)
        self.assertAlmostEqual(rec["reductionPercent"],100/3)
        self.assertFalse(rec["runNow"])
        self.assertTrue(all(p["embodiedGCO2e"]==.1 for p in result["periods"]))

    def test_now_and_zero(self):
        for energy,embodied in ((.001,.1),(0,0)):
            r=compare_periods(energy,embodied,parse_forecast(payload(values=(10,)*6),NOW))["recommendation"]
            self.assertTrue(r["runNow"])
            self.assertEqual(r["reductionPercent"],0)

    def test_forecast_only(self):
        periods=parse_forecast(payload(),NOW)
        self.assertEqual(len(periods),6)
        self.assertEqual(periods[0]["forecast"],200)
        self.assertEqual(periods[0]["unit"],"gCO2/kWh")

    def test_missing_period(self):
        p=payload();p["data"].pop(2)
        with self.assertRaises(ForecastError): parse_forecast(p,NOW)

    def test_malformed(self):
        for p in ({}, {"data":None}, {"data":[{}]}):
            with self.assertRaises(ForecastError): parse_forecast(p,NOW)
        for value in (None,-1,True,"100",float("nan")):
            p=payload();p["data"][0]["intensity"]["forecast"]=value
            with self.assertRaises(ForecastError): parse_forecast(p,NOW)

    def test_duplicate(self):
        p=payload();p["data"].append(p["data"][0])
        with self.assertRaises(ForecastError): parse_forecast(p,NOW)

    def test_timezones(self):
        for month,day in ((3,29),(10,25)):
            now=datetime(2026,month,day,0,45,tzinfo=timezone.utc)
            local=now.astimezone(ZoneInfo("Europe/London"))
            self.assertEqual(parse_forecast(payload(now),local),parse_forecast(payload(now),now))
        boundary=NOW.replace(minute=30)
        self.assertEqual(period_start(boundary),boundary)
        p=payload();p["data"][0]["from"]="2026-10-29T10:00:00"
        with self.assertRaises(ForecastError):parse_forecast(p,NOW)

    def test_current_local_slot(self):
        """Keep the half-hour containing now, regardless of the user's timezone."""
        for zone in ("Europe/Rome", "Asia/Shanghai", "America/New_York"):
            local = datetime(2026, 11, 3, 15, 15, tzinfo=ZoneInfo(zone))
            rows = parse_forecast(payload(local), local)
            start = datetime.fromisoformat(rows[0]["start"]).astimezone(ZoneInfo(zone))
            end = datetime.fromisoformat(rows[0]["end"]).astimezone(ZoneInfo(zone))
            self.assertEqual((start.hour, start.minute), (15, 0))
            self.assertEqual((end.hour, end.minute), (15, 30))
            self.assertEqual(len(rows), 6)

    def test_network_errors(self):
        for error in (TimeoutError("timeout"),URLError("offline")):
            with patch("carbon_footprint_estimator.carbon_intensity.urlopen",side_effect=error):
                with self.assertRaises(ForecastError): fetch_forecast(NOW)

    def test_network_parse(self):
        with patch("carbon_footprint_estimator.carbon_intensity.urlopen",return_value=io.BytesIO(json.dumps(payload()).encode())) as call:
            self.assertEqual(len(fetch_forecast(NOW)),6)
            self.assertEqual(call.call_args.kwargs["timeout"],10)
        with patch("carbon_footprint_estimator.carbon_intensity.urlopen",return_value=io.BytesIO(b"bad")):
            with self.assertRaises(ForecastError):fetch_forecast(NOW)

    def test_partial_result(self):
        def fail(now):raise ForecastError("offline")
        r=estimate(100,50,energy_fn=lambda a,b:dict(energyKWh=.001,embodiedGCO2e=.1),forecast_fn=fail)
        self.assertEqual(r["energyKWh"],.001)
        self.assertIsNone(r["recommendation"])
        self.assertEqual(r["forecastError"],"offline")

    def test_shared_environment(self):
        """Use EcoLogits directly, without starting another Python interpreter."""
        with patch("carbon_footprint_estimator.ecologits_adapter.calculate_local", return_value={"energyWh": 0}) as calculate:
            self.assertEqual(estimate_energy(10, 20), {"energyWh": 0})
            calculate.assert_called_once_with(10, 20)

    def test_ecologits_real(self):
        r=estimate_energy(142,152)
        self.assertEqual(r["inputTokens"],142)
        self.assertEqual(r["outputTokens"],152)
        self.assertAlmostEqual(r["energyWh"],.012645621842727325)
        self.assertAlmostEqual(r["energyWh"],1000*r["energyKWh"])
        self.assertAlmostEqual(r["energyKWh"],1.2*sum(r["allocation"].values()))
        expected=r["durationSeconds"]*(5700/8+273)/(94608000*64)*1000
        self.assertAlmostEqual(r["embodiedGCO2e"],expected)
        self.assertEqual(r["scenario"]["requiredGPUs"],1)
        self.assertEqual(estimate_energy(999,152)["energyWh"],r["energyWh"])
        self.assertEqual(estimate_energy(100,0)["energyWh"],0)

    def test_fixed_pipeline(self):
        r=estimate(142,152,now=NOW,forecast_fn=lambda now:parse_forecast(payload(),now))
        self.assertEqual(r["recommendation"]["start"],r["periods"][2]["start"])
        self.assertAlmostEqual(r["periods"][0]["totalGCO2eApprox"],.003556655644734576)

    def test_frontend_render(self):
        import os, shutil, subprocess
        from pathlib import Path
        node=os.environ.get("PROMPTCARBON_TEST_NODE") or shutil.which("node")
        if not node:self.skipTest("Node is optional for frontend checks.")
        js=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8');
const ids=new Set([...html.matchAll(/id="([^"]+)"/g)].map(x=>x[1]));
function make(tag){return {tag,children:[],value:'',hidden:false,disabled:false,textContent:'',dataset:{},listeners:{},appendChild(x){this.children.push(x)},replaceChildren(){this.children=[]},setAttribute(k,v){this[k]=v},addEventListener(t,f){(this.listeners[t]??=[]).push(f)}}}
const nodes={};function element(id){assert(ids.has(id),id);if(!nodes[id]){nodes[id]=make(id);nodes[id].value=id==='lengthType'?'none':id==='constraintMode'?'manual':id==='targetModel'?'Qwen/Qwen2.5-3B-Instruct':'';}return nodes[id]}
const ctx=vm.createContext({document:{getElementById:element,createElement:make},performance:{now:()=>0},Intl,Date,console});
vm.runInContext(html.match(/<script\b[^>]*>([\s\S]*?)<\/script>/)[1],ctx);
const rows=Array.from({length:6},(_,i)=>({start:new Date(Date.UTC(2026,9,29,10,i*30)).toISOString(),end:new Date(Date.UTC(2026,9,29,10,(i+1)*30)).toISOString(),forecast:100,operationalGCO2eApprox:.1,embodiedGCO2e:.01,totalGCO2eApprox:.11}));
ctx.showCarbon({inputTokens:10,outputTokens:20,energyWh:.1,embodiedGCO2e:.01,periods:rows,recommendation:{...rows[0],runNow:true,reductionGCO2eApprox:0,reductionPercent:0}});
function descendants(node){return node.children.flatMap(x=>[x,...descendants(x)])}
assert.equal(descendants(element('carbonPanel')).find(x=>x.tag==='tbody').children.length,6);
rows.forEach((p,i)=>p.totalGCO2eApprox=(i+1)/100);
ctx.showCarbon({inputTokens:10,outputTokens:20,energyWh:.1,embodiedGCO2e:.01,periods:rows,recommendation:{...rows[0],runNow:true,reductionGCO2eApprox:0,reductionPercent:0}});
const painted=descendants(element('carbonPanel')).find(x=>x.tag==='tbody').children;
for(const color of ['green','medium','red']) assert.equal(painted.filter(x=>x.className==='slot-'+color).length,2);
assert(ids.has('estimateMethod'));
assert(!descendants(element('carbonPanel')).some(x=>x.tag==='details'));
ctx.showCarbon({inputTokens:10,outputTokens:20,energyWh:.1,embodiedGCO2e:.01,forecastError:'offline'});
assert(!descendants(element('carbonPanel')).some(x=>x.tag==='table'));
assert(descendants(element('carbonPanel')).some(x=>x.textContent==='offline'));
ctx.invalidateResults();assert(element('carbonPanel').hidden);
"""
        result=subprocess.run([node,"-e",js,str(Path(__file__).resolve().parents[1]/"index.html")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_http_endpoint(self):
        import server, threading
        from http.server import HTTPServer
        from urllib.request import Request, urlopen
        http=HTTPServer(("127.0.0.1",0),server.Handler)
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        try:
            body=dict(prompt="Return only the word READY.",targetModel="Qwen/Qwen2.5-3B-Instruct")
            # Match the live clock for the API client while keeping all forecasts mocked.
            with patch("carbon_footprint_estimator.carbon_intensity.urlopen",return_value=io.BytesIO(json.dumps(payload(datetime.now(timezone.utc))).encode())):
                request=Request("http://127.0.0.1:"+str(http.server_port)+"/api/estimate-carbon",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                with urlopen(request) as response: result=json.load(response)
            self.assertEqual(len(result["carbonFootprint"]["periods"]),6)
            self.assertEqual(result["outputTokens"],result["carbonFootprint"]["outputTokens"])
            html=(server.ROOT/"index.html").read_text()
            self.assertIn('id="carbonPanel"',html)
            self.assertIn('id="carbonButton"',html)
        finally:
            http.shutdown();http.server_close();thread.join()

    def test_server_pipeline(self):
        import server
        r=server.public_output_response(server.estimate_output_tokens(dict(prompt="Write a short email thanking a colleague for reviewing my report.",targetModel="Qwen/Qwen2.5-3B-Instruct")))
        result=estimate(r["inputTokens"],r["outputTokens"],now=NOW,forecast_fn=lambda now:parse_forecast(payload(),now))
        self.assertEqual(result["outputTokens"],r["outputTokens"])
        self.assertGreater(result["energyWh"],0)
        print("CFE_SAMPLE",json.dumps(dict(analysis=r,carbon=result)))


if __name__ == "__main__":
    unittest.main()
