'use strict';
const palette=['#64b5ff','#ffc273','#b69cff','#65dab0','#ff8fa6','#bada78','#63d8e6'];
const metrics = [
  {id:'accuracy',title:'Imitation accuracy',tag:'1 Imitation/Accuracy',note:'Top-1 and top-3 agreement with logged actions',percent:true,card:true,components:[['Top-1','Accuracy'],['Top-3','Top-3 accuracy']],prefix:'1 Imitation/'},
  {id:'board',title:'Board losses',tag:'2 Board representation/Combined loss',note:'Fraction of starting loss · lower is better',card:true,components:[
    ['Shanten','Shanten'],['Hand-end placement','Hand-end placement'],['Final placement','Final placement'],['Ukeire','Ukeire'],['Yaku','Yaku'],['Opponent hand','Opponent hand']
  ],prefix:'2 Board representation/'},
  {id:'critic',title:'Critic losses',tag:'3 Critic/Combined loss (before 0.1 weight)',note:'Fraction of starting loss · unweighted components',card:true,components:[
    ['Payments','Payment NLL'],['Points','Point MSE'],['Immediate risk','Immediate risk'],['Bonuses','Bonus NLL'],['Scenarios','Scenario NLL'],['Han / fu','Han-fu NLL'],['Discard ranking','Discard ranking']
  ],prefix:'3 Critic/'},
];
const data = {};
const format = (v,m) => v == null ? '—' : m.percent ? (v*100).toFixed(2)+'%' : v.toFixed(4);
const latest = rows => rows.length ? rows[rows.length-1] : null;
for (const m of metrics) {
  const panel = document.createElement('article'); panel.className='chart';
  panel.innerHTML=`<h2>${m.title}</h2><p>${m.note}</p><div class="plot" id="plot-${m.id}"><svg viewBox="0 0 600 245" role="img" aria-label="${m.title} by update"></svg><div class="tooltip"></div></div>`;
  if(m.components){
    const controls=document.createElement('div');controls.className='component-controls';
    controls.innerHTML=`<div class="component-legend">${m.components.map(([label],i)=>`<label style="color:${palette[i]}"><input type="checkbox" data-component="${i}" checked> ${label}</label>`).join('')}</div>`;
    panel.insertBefore(controls,panel.querySelector('.plot'));
    controls.onchange=()=>{if(data[m.id])draw(m);};
  }
  document.querySelector('#charts').append(panel);
  if(m.card) {
    const card=document.createElement('article');card.className='card';
    card.innerHTML=`<h2>${m.title}</h2><div class="number" id="value-${m.id}">—</div><small id="detail-${m.id}">Waiting for metrics</small>`;
    document.querySelector('#cards').append(card);
  }
}
function draw(m) {
  const source=data[m.id];
  const normalized=!m.percent;
  const curves=m.components?m.components.flatMap(([label],i)=>{
    if(!document.querySelector(`#plot-${m.id}`).parentElement.querySelector(`[data-component="${i}"]`).checked)return [];
    const first=source.components[i].train.find(p=>p[2]!==0)||source.components[i].validation.find(p=>p[2]!==0);
    const scale=normalized&&first?Math.abs(first[2]):1;
    return ['train','validation'].map(split=>({label:label+' · '+split,rows:source.components[i][split],color:palette[i],validation:split==='validation',scale}));
  }):['train','validation'].map((split,i)=>({label:i?'Validation':'Training',rows:source[split],color:i?'#ffc273':'#64b5ff',validation:!!i,scale:1}));
  const end=Math.max(...curves.map(c=>latest(c.rows)?.[1]||0),1);
  const start=document.querySelector('#range').value==='recent'?Math.max(0,end-5000):0;
  for(const curve of curves){
    let smoothed;
    curve.points=curve.rows.map(p=>{
      smoothed=smoothed==null?p[2]:.85*smoothed+.15*p[2];
      return [p[0],p[1],(!curve.validation&&document.querySelector('#smooth').checked?smoothed:p[2])/curve.scale];
    }).filter(p=>p[1]>=start);
  }
  const values=curves.flatMap(c=>c.points.map(p=>p[2])).filter(Number.isFinite);
  const svg=document.querySelector(`#plot-${m.id} svg`);
  const width=svg.clientWidth,height=svg.clientHeight;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  if(!values.length){svg.innerHTML=`<text x="${width/2}" y="${height/2}" fill="#9aaac0" text-anchor="middle" font-size="11">No measurements in this range yet</text>`;return;}
  let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max((hi-lo)*.12,Math.abs(hi)*.005,.0001);lo-=pad;hi+=pad;
  const x=step=>60+(step-start)/(end-start||1)*(width-80);
  const y=value=>height-22-(value-lo)/(hi-lo)*(height-32);
  let content='';
  for(let i=0;i<5;i++){
    const value=lo+(hi-lo)*i/4,yy=y(value);
    content+=`<line x1="60" x2="${width-20}" y1="${yy}" y2="${yy}" stroke="#283447"/><text x="50" y="${yy+4}" fill="#9aaac0" font-size="11" text-anchor="end">${m.percent?(value*100).toFixed(1)+'%':normalized?value.toFixed(2)+'×':value.toFixed(3)}</text>`;
  }
  for(let i=0;i<4;i++){const step=start+(end-start)*i/3;content+=`<text x="${x(step)}" y="${height-5}" fill="#9aaac0" font-size="11" text-anchor="middle">${Math.round(step).toLocaleString()}</text>`;}
  for(const curve of curves){
    content+=`<path d="${curve.points.map((p,i)=>`${i?'L':'M'}${x(p[1]).toFixed(2)},${y(p[2]).toFixed(2)}`).join(' ')}" fill="none" stroke="${curve.color}" stroke-width="2" ${curve.validation?'stroke-dasharray="5 4"':''}/>`;
    if(curve.validation||curve.points.length===1)for(const p of curve.points)content+=`<circle cx="${x(p[1])}" cy="${y(p[2])}" r="3" fill="${curve.color}"/>`;
  }
  svg.innerHTML=content;
  svg.onpointermove=e=>{
    const rect=svg.getBoundingClientRect(),step=start+((e.clientX-rect.left)/rect.width*width-60)/(width-80)*(end-start);
    document.querySelector(`#plot-${m.id} .tooltip`).textContent=curves.filter(c=>c.rows.length).map(c=>{
      const p=c.rows.reduce((a,b)=>Math.abs(a[1]-step)<Math.abs(b[1]-step)?a:b);
      return `${c.label} @ ${p[1].toLocaleString()}: ${format(p[2],m)}${normalized?' ('+(p[2]/c.scale).toFixed(2)+'×)':''}`;
    }).join('\n');
  };
  svg.onpointerleave=()=>{document.querySelector(`#plot-${m.id} .tooltip`).textContent='';};
}
async function get(path){const r=await fetch(path,{cache:'no-store'});if(r.status===404&&path.includes('/scalars?'))return [];if(!r.ok)throw new Error(`Dashboard data unavailable (${r.status}). Retrying…`);return r.json();}
async function refresh(){
  try{
    const [run,speed,...series]=await Promise.all([
      get('/tensorboard/status.json'),
      get('/tensorboard/charts/data/plugin/scalars/scalars?run=.&tag='+encodeURIComponent('4 Throughput/positions_per_second')),
      ...metrics.map(async m=>{
        const pair=async tag=>{const [train,validation]=await Promise.all(['/train','/validation'].map(s=>get('/tensorboard/charts/data/plugin/scalars/scalars?run=.&tag='+encodeURIComponent(tag+s))));return {train,validation:validation||[]};};
        return {...await pair(m.tag),components:m.components?await Promise.all(m.components.map(([,tag])=>pair(m.prefix+tag))):[]};
      })
    ]);
    let index=0;
    for(const m of metrics){
      data[m.id]=series[index++];
      const last=latest(data[m.id].train),val=latest(data[m.id].validation);
      if(m.card){
        document.querySelector('#value-'+m.id).textContent=m.percent?`${format(last?.[2],m)} / ${(latest(data[m.id].components[1].train)?format(latest(data[m.id].components[1].train)[2],m):'pending')}`:format(last?.[2],m);
        document.querySelector('#detail-'+m.id).textContent=m.percent?`Top-1 / top-3 · validation ${format(val?.[2],m)} / ${format(latest(data[m.id].components[1].validation)?.[2],m)}`:`Combined · validation ${format(val?.[2],m)}${val?' at update '+val[1].toLocaleString():''}`;
      }
      draw(m);
    }
    const step=run.status.updates??Math.max(...metrics.map(m=>latest(data[m.id].train)?.[1]||0)),total=Object.values(run.plan.phase_updates).reduce((a,b)=>a+b,0);
    document.querySelector('#state').textContent=({validation:'Validating',complete:'Complete',initializing:'Initializing',preflight:'Preflight',rebuild:'Rebuilding records','throughput-natural':'Checking natural throughput','throughput-focused':'Checking focused throughput',failed:'Failed',stopped:'Stopped'})[run.status.stage]||'Training';
    document.querySelector('#progress').textContent=`${(100*step/total).toFixed(2)}% complete · update ${step.toLocaleString()}`;
    const phases=document.querySelector('#phases');phases.replaceChildren();let offset=0;
    for(const [name,count] of Object.entries(run.plan.phase_updates)){
      const segment=document.createElement('div');segment.className='segment';segment.style.width=(100*count/total)+'%';segment.title=name;
      const fill=document.createElement('span');fill.style.width=(100*Math.min(1,Math.max(0,(step-offset)/count)))+'%';segment.append(fill);phases.append(segment);offset+=count;
    }
    document.querySelector('#exposures').textContent=`${(step*run.plan.effective_batch/1e6).toFixed(1)}M / ${(run.plan.position_exposures/1e9).toFixed(2)}B position exposures`;
    document.querySelector('#throughput').textContent=latest(speed)?`${(latest(speed)[2]/1000).toFixed(1)}k positions/sec${run.status.stage==='initializing'?' (last active)':''}`:'';
    const speeds=speed.slice(-5).map(p=>p[2]).sort((a,b)=>a-b);
    document.querySelector('#eta').textContent=run.status.stage==='initializing'?'Restoring data position · top-3 pending':speeds.length?`~${((total-step)*run.plan.effective_batch/speeds[Math.floor(speeds.length/2)]/3600).toFixed(1)}h remaining at recent speed`:'';
    document.querySelector('#updated').textContent='Updated '+new Date().toLocaleTimeString()+' · refreshes every 15 seconds';
    document.querySelector('#error').textContent='';
  }catch(e){document.querySelector('#error').textContent=e.message;}
  setTimeout(refresh,15000);
}
for(const id of ['smooth','range'])document.querySelector('#'+id).onchange=()=>metrics.filter(m=>data[m.id]).forEach(draw);
window.addEventListener('resize',()=>metrics.filter(m=>data[m.id]).forEach(draw));
refresh();
