/* Regular VAE density/importance sweep with paired Gram diagnostics. */
(() => {
  "use strict";
  const S = { experimentId: null, summaries: [], cell: null, config: null };
  const q = (selector, root = document) => root.querySelector(selector);
  const qa = (selector, root = document) => [...root.querySelectorAll(selector)];
  const fmt = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

  function markup() {
    return `<div class="gram-shell">
      <div class="gram-hero">
        <div><div class="gram-eyebrow">Regular VAE · orthogonality study</div><h1>Linear vs manifold Gram experiment</h1>
          <p>Train the exact density × importance × ambient-dimension sweep on angular data, then compare Euclidean weight alignment with alignment under the decoder-induced metric.</p></div>
        <div class="gram-run-state"><span>Experiment state</span><strong id="gram-state">Ready · 616 VAE fits</strong></div>
      </div>
      <div class="gram-config">
        <section class="gram-card"><h2>Locked experimental design</h2><p>The latent bottleneck stays fixed at 2 while the higher, observed dimension is swept.</p>
          <div class="gram-fixed-grid">
            <div class="gram-fixed"><span>Density</span><strong>0–100% · step 10</strong></div>
            <div class="gram-fixed"><span>Importance decay</span><strong>0–3 · step 0.5</strong></div>
            <div class="gram-fixed"><span>Ambient dimension</span><strong>3–10 · step 1</strong></div>
            <div class="gram-fixed"><span>Hidden dimension</span><strong>2 · fixed</strong></div>
          </div>
      <div class="gram-formula">Lᵢⱼ = vᵢᵀvⱼ · L̂ᵢⱼ = |Lᵢⱼ|/(‖vᵢ‖‖vⱼ‖)<br>G(z) = J(z)ᵀJ(z) · MGᵢⱼ = Eₓ E<sub>z∼q(z|x)</sub>[vᵢᵀG(z)vⱼ]</div>
        </section>
        <section class="gram-card"><h2>Training & geometry</h2><p>Density is Bernoulli coordinate sparsity applied after sampling the angular curve.</p>
          <div class="gram-form">
            <div class="gram-field full"><label>Geometry</label><select id="gram-geometry"><option value="circle">Circle · r(θ)=1</option><option value="custom">Custom radial equation</option></select></div>
            <div class="gram-field full gram-custom hidden"><label>r(θ) equation</label><input id="gram-equation" value="1 + 0.25*cos(3*theta)" spellcheck="false"></div>
            <div class="gram-field"><label>Steps per VAE</label><input id="gram-steps" type="number" min="1" max="5000" value="200"></div>
            <div class="gram-field"><label>Batch size</label><input id="gram-batch" type="number" min="8" max="4096" value="128"></div>
            <div class="gram-field"><label>Learning rate</label><input id="gram-lr" type="number" min="0.00001" max="0.1" step="0.0001" value="0.003"></div>
            <div class="gram-field"><label>Seed</label><input id="gram-seed" type="number" value="42"></div>
            <label class="gram-toggle full"><input id="gram-unit-norm" type="checkbox"><span><strong>Unit-normalize W<sub>&mu;</sub> feature weights</strong><small>Project every tied mean/decoder column to unit L2 norm after each optimizer step.</small></span></label>
          </div>
          <div class="gram-actions"><button class="gram-button" id="gram-run">Run exact sweep</button><span class="gram-note">The full 616-model run can take several minutes on CPU.</span></div>
          <div class="gram-progress hidden" id="gram-progress"></div>
        </section>
      </div>
      <div class="gram-results hidden" id="gram-results">
        <section class="gram-card gram-toolbar">
          <div class="gram-field"><label>Ambient dimension</label><select id="gram-dimension"></select></div>
          <div class="gram-field"><label>Density</label><select id="gram-density"></select></div>
          <div class="gram-field"><label>Importance decay</label><select id="gram-importance"></select></div>
          <button class="gram-button secondary" id="gram-export">Export summary JSON</button>
        </section>
        <div class="gram-stats">
          <div class="gram-stat"><span>Final loss</span><strong id="gram-stat-loss">—</strong></div>
          <div class="gram-stat good"><span>Linear off-diagonal</span><strong id="gram-stat-linear">—</strong></div>
          <div class="gram-stat good"><span>Manifold off-diagonal</span><strong id="gram-stat-manifold">—</strong></div>
          <div class="gram-stat warn"><span>Manifold − linear</span><strong id="gram-stat-delta">—</strong></div>
          <div class="gram-stat"><span>Jacobian condition</span><strong id="gram-stat-condition">—</strong></div>
          <div class="gram-stat"><span>Volume factor μ ± σ</span><strong id="gram-stat-factor">—</strong></div>
        </div>
        <section class="gram-card" style="margin-bottom:14px"><div class="gram-panel-head"><div><h2>Raw Gram matrices</h2><p>Signed inner products before diagonal normalization.</p></div><span>same selected VAE</span></div>
          <div class="gram-compare"><div class="gram-panel"><div class="gram-panel-head"><h3>Linear · WᵀW</h3><span>Euclidean</span></div><div class="gram-canvas-wrap"><canvas id="gram-linear-raw"></canvas></div></div>
          <div class="gram-panel"><div class="gram-panel-head"><h3>Manifold · WᵀḠW</h3><span>decoder metric</span></div><div class="gram-canvas-wrap"><canvas id="gram-manifold-raw"></canvas></div></div></div>
        </section>
        <section class="gram-card" style="margin-bottom:14px"><div class="gram-panel-head"><div><h2>Normalized orthogonality</h2><p>Absolute cosine-like alignment; zero off-diagonal is orthogonal and one is collinear.</p></div><span>shared 0 → 1 scale</span></div>
          <div class="gram-compare"><div class="gram-panel"><div class="gram-panel-head"><h3>Linear normalized</h3><span>|vᵢᵀvⱼ|/(‖vᵢ‖‖vⱼ‖)</span></div><div class="gram-canvas-wrap"><canvas id="gram-linear-normalized"></canvas></div></div>
          <div class="gram-panel"><div class="gram-panel-head"><h3>Manifold normalized</h3><span>|MGᵢⱼ|/√(MGᵢᵢMGⱼⱼ)</span></div><div class="gram-canvas-wrap"><canvas id="gram-manifold-normalized"></canvas></div></div></div>
        </section>
        <section class="gram-diagnostics">
      <div class="gram-card"><div class="gram-panel-head"><div><h2>Jacobian & metric factor</h2><p>Posterior-sampled mean decoder Jacobian and Ḡ = E[JᵀJ]. The factor is E[√det(G(z))].</p></div></div><div class="gram-diagnostic-grid"><canvas id="gram-jacobian"></canvas><canvas id="gram-metric"></canvas></div></div>
          <div class="gram-card"><div class="gram-panel-head"><div><h2>Orthogonality across density</h2><p>Mean normalized off-diagonal alignment for the selected dimension and importance.</p></div></div><div class="gram-canvas-wrap"><canvas id="gram-trend"></canvas></div></div>
        </section>
      </div>
    </div>`;
  }

  function setState(text) { const el = q("#gram-state"); if (el) el.textContent = text; }
  function notify(message, type = "info") {
    if (typeof window.showToast === "function") window.showToast(message, type);
    else console[type === "error" ? "error" : "log"](message);
  }
  async function api(path, options = {}) {
    const response = await fetch(path, { headers: {"Content-Type":"application/json"}, ...options });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function unique(key) { return [...new Set(S.summaries.map(item => item[key]))].sort((a,b) => a-b); }
  function fillSelect(selector, values, label) {
    const select = q(selector); select.innerHTML = values.map(value => `<option value="${value}">${label(value)}</option>`).join("");
  }
  function selectedSummary() {
    const dimension = Number(q("#gram-dimension").value), density = Number(q("#gram-density").value), decay = Number(q("#gram-importance").value);
    return S.summaries.find(item => item.ambient_dimension === dimension && Math.abs(item.density-density) < 1e-8 && Math.abs(item.importance_decay-decay) < 1e-8);
  }

  async function loadSelected() {
    const summary = selectedSummary(); if (!summary) return;
    setState(`Loading cell ${summary.index + 1} of ${S.summaries.length}…`);
    try {
      const data = await api(`/api/orthogonality/experiments/${encodeURIComponent(S.experimentId)}/cell?index=${summary.index}`);
      S.cell = data.cell; renderCell(); setState(`${S.summaries.length} fits · cell ${summary.index + 1} selected`);
    } catch (error) { notify(error.message, "error"); setState("Cell load failed"); }
  }

  function setupResults() {
    fillSelect("#gram-dimension", unique("ambient_dimension"), value => `${value} observed features`);
    fillSelect("#gram-density", unique("density"), value => `${Math.round(value*100)}%`);
    fillSelect("#gram-importance", unique("importance_decay"), value => Number(value).toFixed(1));
    q("#gram-results").classList.remove("hidden");
    renderCell();
  }

  function canvasSize(canvas) {
    const ratio = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(320, Math.floor(rect.width * ratio)); canvas.height = Math.max(240, Math.floor(rect.height * ratio));
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio,0,0,ratio,0,0); return {ctx, width:rect.width, height:rect.height};
  }
  function color(value, maxAbs, normalized) {
    if (normalized) { const t=Math.max(0,Math.min(1,value)); return `rgb(${Math.round(12+230*t)},${Math.round(20+130*(1-t))},${Math.round(38+160*(1-t))})`; }
    const t=Math.max(-1,Math.min(1,value/(maxAbs||1))); return t>=0 ? `rgb(${Math.round(18+220*t)},${Math.round(28+70*(1-t))},${Math.round(48+65*(1-t))})` : `rgb(${Math.round(20+40*(1+t))},${Math.round(35+120*(-t))},${Math.round(55+190*(-t))})`;
  }
  function heatmap(selector, matrix, title, normalized = false) {
    const canvas=q(selector); if(!canvas||!matrix?.length)return; const {ctx,width,height}=canvasSize(canvas); ctx.clearRect(0,0,width,height);
    const rows=matrix.length, cols=matrix[0].length, margin={l:42,r:18,t:35,b:35}, areaW=width-margin.l-margin.r, areaH=height-margin.t-margin.b, cell=Math.min(areaW/cols,areaH/rows), gridW=cell*cols, gridH=cell*rows, x0=margin.l+(areaW-gridW)/2, y0=margin.t+(areaH-gridH)/2;
    const maxAbs=normalized?1:Math.max(...matrix.flat().map(Math.abs),1e-9); ctx.fillStyle="#c4bdd8";ctx.font="600 12px system-ui";ctx.fillText(title,12,18);
    matrix.forEach((row,i)=>row.forEach((value,j)=>{ctx.fillStyle=color(value,maxAbs,normalized);ctx.fillRect(x0+j*cell,y0+i*cell,cell+.4,cell+.4);if(Math.max(rows,cols)<=10&&cell>28){ctx.fillStyle=Math.abs(normalized?value:value/maxAbs)>.55?"#fff":"#d8d2e8";ctx.font=`${Math.min(11,cell*.25)}px monospace`;ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(Number(value).toFixed(2),x0+(j+.5)*cell,y0+(i+.5)*cell);}}));
    const isJacobian=cols===2&&rows!==2;ctx.fillStyle="#8f87a8";ctx.font="10px monospace";ctx.textAlign="center";ctx.textBaseline="top";for(let i=0;i<cols;i++)ctx.fillText(isJacobian?`z${i+1}`:`v${i+1}`,x0+(i+.5)*cell,y0+gridH+6);ctx.textAlign="right";ctx.textBaseline="middle";for(let i=0;i<rows;i++)ctx.fillText(isJacobian?`x${i+1}`:`v${i+1}`,x0-7,y0+(i+.5)*cell);
  }
  function trend() {
    const canvas=q("#gram-trend"); if(!canvas)return; const {ctx,width,height}=canvasSize(canvas), dim=Number(q("#gram-dimension").value), decay=Number(q("#gram-importance").value);
    const rows=S.summaries.filter(x=>x.ambient_dimension===dim&&Math.abs(x.importance_decay-decay)<1e-8).sort((a,b)=>a.density-b.density), m={l:42,r:18,t:28,b:38};ctx.clearRect(0,0,width,height);ctx.strokeStyle="#312d43";ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=m.t+(height-m.t-m.b)*i/4;ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(width-m.r,y);ctx.stroke();ctx.fillStyle="#817997";ctx.font="10px monospace";ctx.textAlign="right";ctx.fillText((1-i/4).toFixed(2),m.l-7,y+3)}
    const draw=(key,stroke)=>{ctx.strokeStyle=stroke;ctx.lineWidth=2.2;ctx.beginPath();rows.forEach((row,i)=>{const x=m.l+row.density*(width-m.l-m.r),y=m.t+(1-Math.min(1,row[key]))*(height-m.t-m.b);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();};draw("linear_offdiag","#f59e0b");draw("manifold_offdiag","#ec4899");
    ctx.fillStyle="#f59e0b";ctx.textAlign="left";ctx.fillText("Linear",m.l,15);ctx.fillStyle="#ec4899";ctx.fillText("Manifold",m.l+55,15);ctx.fillStyle="#817997";ctx.textAlign="center";ctx.fillText("density 0% → 100%",(m.l+width-m.r)/2,height-10);
  }
  function renderCell() {
    const c=S.cell;if(!c)return;q("#gram-stat-loss").textContent=fmt(c.loss);q("#gram-stat-linear").textContent=fmt(c.linear_offdiag);q("#gram-stat-manifold").textContent=fmt(c.manifold_offdiag);q("#gram-stat-delta").textContent=`${c.orthogonality_delta>=0?"+":""}${fmt(c.orthogonality_delta)}`;q("#gram-stat-condition").textContent=fmt(c.jacobian_condition,3);q("#gram-stat-factor").textContent=`${fmt(c.factor_mean,3)} ± ${fmt(c.factor_std,3)}`;
    heatmap("#gram-linear-raw",c.linear_gram,"signed linear Gram");heatmap("#gram-manifold-raw",c.manifold_gram,"signed manifold Gram");heatmap("#gram-linear-normalized",c.linear_normalized,"linear orthogonality",true);heatmap("#gram-manifold-normalized",c.manifold_normalized,"manifold orthogonality",true);heatmap("#gram-jacobian",c.jacobian,"mean J(z)");heatmap("#gram-metric",c.metric,"mean G(z)");trend();
  }

  async function run() {
    const button=q("#gram-run");button.disabled=true;q("#gram-progress").classList.remove("hidden");setState("Training 616 regular VAEs…");
    const payload={geometry:q("#gram-geometry").value,custom_equation:q("#gram-equation").value,steps:Number(q("#gram-steps").value),batch_size:Number(q("#gram-batch").value),learning_rate:Number(q("#gram-lr").value),seed:Number(q("#gram-seed").value),unit_norm_weights:q("#gram-unit-norm").checked};
    try{const data=await api("/api/orthogonality/run",{method:"POST",body:JSON.stringify(payload)});S.experimentId=data.experiment_id;S.summaries=data.summaries;S.cell=data.cell;S.config=data.config;setupResults();setState(`${S.summaries.length} fits complete · ${data.device}${data.config.unit_norm_weights?" · unit-norm W_mu":""}`);notify("Gram experiment completed.","success");}
    catch(error){notify(error.message,"error");setState("Experiment failed");}finally{button.disabled=false;q("#gram-progress").classList.add("hidden");}
  }
  function exportSummary(){if(!S.summaries.length)return;const blob=new Blob([JSON.stringify({experiment_id:S.experimentId,config:S.config,summaries:S.summaries},null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`${S.experimentId}-summary.json`;a.click();URL.revokeObjectURL(url);}
  function init(){const workspace=q("#workspace-orthogonality");if(!workspace)return;workspace.innerHTML=markup();q("#gram-geometry").addEventListener("change",e=>q(".gram-custom").classList.toggle("hidden",e.target.value!=="custom"));q("#gram-run").addEventListener("click",run);["#gram-dimension","#gram-density","#gram-importance"].forEach(sel=>q(sel).addEventListener("change",loadSelected));q("#gram-export").addEventListener("click",exportSummary);let timer;window.addEventListener("resize",()=>{clearTimeout(timer);timer=setTimeout(renderCell,120)});}
  document.addEventListener("DOMContentLoaded",init);
})();
