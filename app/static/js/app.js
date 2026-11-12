function pollJob(){
  const box = document.getElementById('job');
  if(!box) return;
  const id = box.dataset.job;
  // Status the page was first rendered with. If it already loaded in a terminal
  // state, we must not trigger any reload (that would loop).
  const initialStatus = (document.getElementById('status')?.textContent || '').trim();
  let misses = 0;
  const set = (el, v) => { const n = document.getElementById(el); if(n) n.textContent = v; };

  function renderSummary(summary){
    const body = document.getElementById('summary-body');
    if(!body) return;
    if(!summary || !summary.length){
      body.innerHTML = '<tr><td colspan="2" class="empty">No summary yet.</td></tr>';
      return;
    }
    body.innerHTML = summary.map(s =>
      '<tr><td>' + (s.species ?? '') + '</td><td>' + (s.count ?? 0) + '</td></tr>'
    ).join('');
  }

  async function tick(){
    try{
      const r = await fetch('/api/jobs/' + encodeURIComponent(id));
      if(r.status === 404){
        if(++misses < 5){ setTimeout(tick, 1500); }
        return;
      }
      const j = await r.json();
      if(j && j.id){
        misses = 0;
        const statusEl = document.getElementById('status');
        if(statusEl){
          statusEl.textContent = j.status || '';
          statusEl.className = 'pill ' + (j.status || '');
        }
        set('msg', j.message || '');
        set('processed', j.processed || 0);
        set('total', j.total || 0);
        set('detections', j.detections || 0);
        set('failed', j.failed || 0);
        set('copied', j.copied || 0);
        const bar = document.getElementById('bar');
        if(bar) bar.style.width = (j.progress || 0) + '%';
        renderSummary(j.summary);
        const stopBtn = document.getElementById('stop-btn');
        if(stopBtn && j.status === 'stopping'){ stopBtn.disabled = true; stopBtn.textContent = 'Stopping…'; }
        // Keep polling through 'finalizing' and 'stopping'. Stop on terminal states.
        const terminal = ['complete','error','stopped'];
        if(!terminal.includes(j.status)){
          setTimeout(tick, 1500);
        } else if(!terminal.includes(initialStatus)){
          // The job ended while we watched. Reload once so the server renders
          // the right buttons (Resume, no Stop). If the page already loaded
          // in a terminal state, do nothing (that would loop).
          setTimeout(() => location.reload(), 800);
        }
      } else {
        setTimeout(tick, 2000);
      }
    }catch(e){
      setTimeout(tick, 3000);
    }
  }
  tick();
}

// Processing page: show only the fields for the selected pipeline. Hidden
// fields are disabled so they are neither submitted nor validated.
function pipelineToggle(){
  const radios = document.querySelectorAll('input[name="pipeline"]');
  const groups = document.querySelectorAll('[data-pipeline]');
  function apply(){
    const selected = document.querySelector('input[name="pipeline"]:checked')?.value;
    groups.forEach(g => {
      const on = g.dataset.pipeline === selected;
      g.hidden = !on;
      g.querySelectorAll('select,input').forEach(el => { el.disabled = !on; });
    });
  }
  radios.forEach(r => r.addEventListener('change', apply));
  apply();
}

// Processing page: show the copy options only when copying is switched on,
// and fill the species dropdown from the selected model's class list.
function copyToggle(){
  const box = document.getElementById('copy_matches');
  const opts = document.getElementById('copy_options');
  if(!box || !opts) return;
  const apply = () => { opts.hidden = !box.checked; if(box.checked) loadSpecies(); };
  box.addEventListener('change', apply);
  document.querySelectorAll('input[name="pipeline"], select[name="model"], select[name="two_stage_model"]')
    .forEach(el => el.addEventListener('change', () => { if(box.checked) loadSpecies(); }));
  document.getElementById('species_list')?.addEventListener('change', speciesSummary);
  apply();
}

let speciesKey = null;
async function loadSpecies(){
  const list = document.getElementById('species_list');
  if(!list) return;
  const pipeline = document.querySelector('input[name="pipeline"]:checked')?.value || 'single_stage';
  const params = new URLSearchParams({pipeline});
  const sel = document.querySelector(pipeline === 'two_stage' ? 'select[name="two_stage_model"]' : 'select[name="model"]');
  params.set(pipeline === 'two_stage' ? 'two_stage_model' : 'model', sel ? sel.value : '');
  const key = params.toString();
  if(key === speciesKey) return;  // already showing this model's species
  speciesKey = key;
  const msg = text => { const p = document.createElement('p'); p.className = 'muted'; p.textContent = text; list.replaceChildren(p); };
  msg("Loading the model's species…");
  speciesSummary();
  try{
    const r = await fetch('/api/model-species?' + key);
    const j = await r.json();
    if(key !== speciesKey) return;  // the model was changed again meanwhile
    if(!r.ok || j.error){ speciesKey = null; msg(j.error || 'Could not read the class list.'); return; }
    if(!j.species.length){ msg('This model has no class names, so species cannot be chosen.'); return; }
    list.replaceChildren(...j.species.map(sp => {
      const label = document.createElement('label');
      label.className = 'check';
      const input = document.createElement('input');
      input.type = 'checkbox'; input.name = 'copy_species'; input.value = sp;
      label.append(input, ' ' + sp);
      return label;
    }));
  }catch(e){
    speciesKey = null;
    msg('Could not read the class list.');
  }
  speciesSummary();
}

function speciesSummary(){
  const summary = document.querySelector('#species_picker summary');
  if(!summary) return;
  const chosen = [...document.querySelectorAll('#species_list input:checked')].map(i => i.value);
  summary.textContent = chosen.length ? chosen.join(', ') : 'Any detection';
}
