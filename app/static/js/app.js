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
        const bar = document.getElementById('bar');
        if(bar) bar.style.width = (j.progress || 0) + '%';
        renderSummary(j.summary);
        // Keep polling through 'finalizing'. Stop on terminal states.
        if(!['complete','error'].includes(j.status)){
          setTimeout(tick, 1500);
        } else if(j.status === 'error' && initialStatus !== 'error'){
          // The job errored while we watched. Reload once so the server renders
          // the Resume button. If the page already loaded errored, do nothing.
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
