"""Self-contained interactive HTML 'promo board' for the editor.

Series heat-map (every episode x time, colored by hook/cliff strength, promos
overlaid), ranked cards with the paper-edit transcript, approve/reject/notes
that export feedback.json for `python -m promo_finder learn`.
Contains script text -> keep it internal (input pack is confidential).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .candidates import Segment
from .judge import transcript


def write_board(path: Path, eps, sig, recs, meta, bin_s=4.0):
    E = {e.num: e for e in eps}
    heat = []
    for e in eps:
        n = int(e.duration // bin_s) + 1
        h, c = [0.0] * n, [0.0] * n
        for ev, s in zip(e.events, sig[e.num]):
            k = min(n - 1, int(ev.start // bin_s))
            h[k] = max(h[k], s.hook)
            k2 = min(n - 1, int(max(0, ev.end - 0.01) // bin_s))
            c[k2] = max(c[k2], s.cliff)
        heat.append({"ep": e.num, "dur": round(e.duration, 1), "h": [round(x, 1) for x in h], "c": [round(x, 1) for x in c],
                     "log": e.logline})
    cards = []
    for r in recs:
        segs = [Segment(s["episode"], s["start_s"], s["end_s"]) for s in r["segments"]]
        cards.append({**r, "transcript": transcript(E, segs, max_cam=160)})
    data = json.dumps({"meta": meta, "heat": heat, "promos": cards, "bin": bin_s}, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", data).replace("__TITLE__", html.escape(meta["show"]))
    path.write_text(page)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo Board</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dd;--acc:#c2410c;--hook:#dc2626;--cliff:#2563eb;--ok:#15803d;--bad:#b91c1c;--chip:#f1efe9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--card:#1b1b1f;--ink:#ececf0;--mute:#9a9aa3;--line:#2c2c33;--acc:#fb923c;--hook:#f87171;--cliff:#60a5fa;--ok:#4ade80;--bad:#f87171;--chip:#26262c}}
:root[data-theme="dark"]{--bg:#121214;--card:#1b1b1f;--ink:#ececf0;--mute:#9a9aa3;--line:#2c2c33;--acc:#fb923c;--hook:#f87171;--cliff:#60a5fa;--ok:#4ade80;--bad:#f87171;--chip:#26262c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:20px 16px 8px;max-width:1200px;margin:auto}h1{font-size:22px;margin:0 0 4px}.mute{color:var(--mute)}
.wrap{max-width:1200px;margin:auto;padding:0 16px 60px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.stat b{display:block;font-size:18px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:14px 0;overflow:hidden}
#heat{width:100%;display:block;cursor:crosshair}.legend{display:flex;gap:14px;font-size:12px;color:var(--mute);flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:6px}
.card.approve{outline:2px solid var(--ok)}.card.reject{opacity:.55;outline:2px solid var(--bad)}
.rk{font-weight:700;color:var(--acc);font-size:18px}.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{background:var(--chip);border-radius:999px;padding:2px 8px;font-size:12px}
.seg{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.bar{height:6px;background:var(--chip);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.bars{display:grid;grid-template-columns:70px 1fr 28px;gap:4px 8px;align-items:center;font-size:12px}
details pre{white-space:pre-wrap;font-size:11.5px;background:var(--chip);padding:8px;border-radius:8px;max-height:320px;overflow:auto}
.btns{display:flex;gap:6px;margin-top:auto}button{font:inherit;border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:8px;padding:5px 10px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}textarea{width:100%;font:inherit;font-size:12px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px;resize:vertical}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}select{font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:4px}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);padding:6px 8px;border-radius:8px;font-size:12px;max-width:320px;display:none;z-index:9}
</style></head><body>
<header><h1>Promo Board · __TITLE__</h1><div class="mute" id="sub"></div><div class="stats" id="stats"></div></header>
<div class="wrap">
<div class="panel"><div class="toolbar" style="justify-content:space-between"><b>Whole series at a glance</b>
<div class="legend"><span><i class="sw" style="background:var(--hook)"></i>hook strength</span><span><i class="sw" style="background:var(--cliff)"></i>cliffhanger strength</span><span><i class="sw" style="border:2px solid var(--acc)"></i>picked promo (number = rank)</span></div></div>
<canvas id="heat"></canvas><div class="mute" style="font-size:12px">Each row is one episode (1 at top), left→right is time. Click a box to open that promo.</div></div>
<div class="toolbar"><b>Ranked promos</b><select id="ftype"><option value="">all types</option><option>single</option><option>supercut</option></select>
<span class="mute" id="count"></span><span style="flex:1"></span><button id="exp">Export feedback.json</button></div>
<div class="cards" id="cards" style="margin-top:10px"></div></div><div id="tip"></div>
<script>
const D=__DATA__;const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const M=D.meta;document.getElementById('sub').textContent=`${M.episodes} episodes · model ${M.model} · ${M.candidates} cut-safe candidates scanned · ${M.runtime_s}s`;
const st=[["episodes",M.episodes],["candidates",M.candidates],["LLM calls",M.llm.api_calls],["tokens",(M.llm.tokens_in+M.llm.tokens_out).toLocaleString()],["promos",D.promos.length]];
document.getElementById('stats').innerHTML=st.map(([a,b])=>`<div class="stat"><b>${b}</b><span class="mute">${a}</span></div>`).join('');
let FB={};try{FB=JSON.parse(localStorage.getItem('fb_'+M.show)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('fb_'+M.show,JSON.stringify(FB))}catch(e){}};
const cv=document.getElementById('heat'),tip=document.getElementById('tip');let boxes=[];
function draw(){const W=cv.clientWidth,rowH=Math.max(9,Math.min(14,560/D.heat.length)),padL=34,maxDur=Math.max(...D.heat.map(h=>h.dur));
 const H=rowH*D.heat.length+20;const r=devicePixelRatio||1;cv.width=W*r;cv.height=H*r;cv.style.height=H+'px';const g=cv.getContext('2d');g.scale(r,r);
 const sx=t=>padL+(W-padL-6)*t/maxDur;g.font='10px system-ui';g.fillStyle=css('--mute');
 D.heat.forEach((h,i)=>{const y=i*rowH;if(h.ep%5==0||h.ep==1)g.fillText(h.ep,4,y+rowH-2);
  h.h.forEach((v,k)=>{const a=Math.min(1,v/6);if(a>0.05){g.globalAlpha=a;g.fillStyle=css('--hook');g.fillRect(sx(k*D.bin),y+1,sx(D.bin)-padL+.5,rowH/2-1)}});
  h.c.forEach((v,k)=>{const a=Math.min(1,v/6);if(a>0.05){g.globalAlpha=a;g.fillStyle=css('--cliff');g.fillRect(sx(k*D.bin),y+rowH/2,sx(D.bin)-padL+.5,rowH/2-1)}});
  g.globalAlpha=1;});
 boxes=[];g.lineWidth=2;g.strokeStyle=css('--acc');g.fillStyle=css('--acc');g.font='bold 10px system-ui';
 D.promos.forEach(p=>p.segments.forEach(s=>{const i=D.heat.findIndex(h=>h.ep==s.episode);const x=sx(s.start_s),w=sx(s.end_s)-x,y=i*rowH;
  g.strokeRect(x,y+.5,w,rowH-1);g.fillText(p.rank,x+2,y+rowH-2);boxes.push({x,y,w,h:rowH,p})}));
 g.fillStyle=css('--mute');g.font='10px system-ui';for(let t=0;t<=maxDur;t+=30){g.fillText((t/60|0)+':'+String(t%60).padStart(2,'0'),sx(t),H-4)}}
cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;const hit=boxes.find(o=>x>=o.x&&x<=o.x+o.w&&y>=o.y&&y<=o.y+o.h);
 const row=D.heat[Math.floor(y/(boxes[0]?boxes[0].h:12))];tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
 tip.innerHTML=hit?`<b>#${hit.p.rank} ${hit.p.id}</b><br>${esc(hit.p.hook)}`:(row?`<b>Ep ${row.ep}</b><br>${esc(row.log)}`:'');});
cv.addEventListener('mouseleave',()=>tip.style.display='none');
cv.addEventListener('click',e=>{const b=cv.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;const hit=boxes.find(o=>x>=o.x&&x<=o.x+o.w&&y>=o.y&&y<=o.y+o.h);
 if(hit){const el=document.getElementById('c_'+hit.p.id);el.scrollIntoView({behavior:'smooth',block:'center'});el.animate([{transform:'scale(1.03)'},{transform:'none'}],{duration:500})}});
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function bar(l,v,max=10){if(v==null)return'';const p=Math.max(0,Math.min(100,100*v/max));return`<span>${l}</span><div class="bar"><i style="width:${p}%"></i></div><span>${(+v).toFixed(1)}</span>`}
function render(){const f=document.getElementById('ftype').value;const L=D.promos.filter(p=>!f||p.type==f);document.getElementById('count').textContent=L.length+' shown';
 document.getElementById('cards').innerHTML=L.map(p=>{const s=p.scores,fb=FB[p.id]||{};const segs=p.segments.map(x=>`Ep${x.episode} ${x.start}–${x.end}`).join(' + ');
 return`<div class="card ${fb.verdict||''}" id="c_${p.id}"><div style="display:flex;justify-content:space-between;align-items:baseline"><span class="rk">#${p.rank}</span><span class="chips"><span class="chip">${p.type}</span><span class="chip">${p.duration_s.toFixed(0)}s</span>${p.emotion?`<span class="chip">${esc(p.emotion)}</span>`:''}${p.platform_risk&&p.platform_risk!='none'?`<span class="chip">risk: ${esc(p.platform_risk)}</span>`:''}</span></div>
 <div class="seg">${segs}</div><div><span class="lbl">Hook</span><br>${esc(p.hook)}</div><div><span class="lbl">Cliffhanger</span><br>${esc(p.cliffhanger)}</div>
 <div><span class="lbl">Why</span><br>${esc(p.why)}</div>${p.stranger_summary?`<div><span class="lbl">Stranger test (${p.stranger_match_0_2??'-'}/2)</span><br><i>${esc(p.stranger_summary)}</i></div>`:''}
 <div class="bars">${bar('LLM hook',s.llm_hook)}${bar('LLM cliff',s.llm_cliff)}${bar('standalone',s.llm_standalone)}${bar('rules',s.rules10)}${bar('funnel',s.funnel,1)}</div>
 <details><summary class="mute">Paper edit (what the viewer sees)</summary><pre>${esc(p.transcript)}</pre></details>
 <textarea rows="1" placeholder="editor note…" data-id="${p.id}">${esc(fb.note||'')}</textarea>
 <div class="btns"><button data-v="approve" data-id="${p.id}" class="${fb.verdict=='approve'?'on':''}">✓ Post</button><button data-v="reject" data-id="${p.id}" class="${fb.verdict=='reject'?'on':''}">✗ Skip</button></div></div>`}).join('')}
document.getElementById('cards').addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;const id=b.dataset.id;FB[id]=FB[id]||{};FB[id].verdict=FB[id].verdict==b.dataset.v?'':b.dataset.v;save();render()});
document.getElementById('cards').addEventListener('change',e=>{if(e.target.tagName=='TEXTAREA'){const id=e.target.dataset.id;FB[id]=FB[id]||{};FB[id].note=e.target.value;save()}});
document.getElementById('ftype').onchange=render;
document.getElementById('exp').onclick=()=>{const out={show:M.show,model:M.model,feedback:Object.entries(FB).filter(([k,v])=>v.verdict||v.note).map(([id,v])=>({id,...v}))};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='feedback.json';a.click()};
render();draw();addEventListener('resize',draw);
</script></body></html>"""
