"use strict";
const $ = id => document.getElementById(id);
const state = {tab:"monitor", people:[], token:"", engine:"starting", selected:"", angle:"front", stream:null,
  session:crypto.randomUUID().replaceAll("-",""), busy:false, generation:0, result:null, job:null, pollBusy:false};
const titles = {
  monitor:["Live monitoring","A watchful eye. A safer space.","See who is here and monitor posture, together in one view."],
  enroll:["Enroll a person","Familiar faces, thoughtfully saved.","A name, a few clear angles, and a profile that is ready to recognize."],
  profiles:["Saved profiles","Everyone, at a glance.","See the people you have enrolled and the photos behind each profile."],
  events:["Event history","The moments that need attention.","Review locally recorded posture events and their supporting signals."]};
const angles = {front:"Look straight at the camera.",left:"Turn slightly to your left, keeping both eyes visible.",right:"Turn slightly to your right, keeping both eyes visible.",up:"Lift your chin a little. Keep your face in the frame.",down:"Lower your chin a little. Keep your face in the frame."};
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const initials = name => name.split(/\s+/).map(x=>x[0]).slice(0,2).join("").toUpperCase();
const statusLabel = status => ({ready:"Ready",update_needed:"Update needed",needs_training:"Build needed",needs_photos:"Add photos"}[status] || status);
let toastTimer;
function toast(message,error=false){ $("toast").textContent=message; $("toast").className="toast"+(error?" error":""); $("toast").hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$("toast").hidden=true,6500); }
async function api(path, options={}){
  const headers={"X-SentryCare-Token":state.token,...options.headers};
  if(options.json!==undefined){options.body=JSON.stringify(options.json);headers["Content-Type"]="application/json";delete options.json;}
  const response=await fetch(path,{...options,headers});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||"Request failed");
  return data;
}
function setTab(tab){
  if(!titles[tab])return;
  state.tab=tab; location.hash=tab;
  document.querySelectorAll(".tab-panel").forEach(p=>p.hidden=p.id!==`tab-${tab}`);
  document.querySelectorAll("[data-tab]").forEach(b=>{b.classList.toggle("active",b.dataset.tab===tab);b.setAttribute("aria-current",b.dataset.tab===tab?"page":"false");});
  const [crumb,title,description]=titles[tab];
  $("breadcrumb").textContent=crumb;$("page-title").textContent=title;$("page-description").textContent=description;
  $(tab==="enroll"?"enroll-camera-slot":"monitor-camera-slot").append($("camera-component"));
  if(tab==="events")loadEvents();
  if(tab==="profiles")renderProfiles();
  state.result=null;drawOverlay();updateCapture();
}
function renderPeople(){
  $("stat-profiles").innerHTML=`${state.people.filter(p=>p.trained_count>=3).length} <small>people</small>`;
  $("nav-count").textContent=state.people.length;
  const select=$("person-select");
  select.innerHTML='<option value="">Select a profile</option>'+state.people.map(p=>`<option value="${p.id}">${esc(p.name)} · ${p.sample_count} photos</option>`).join("");
  if(!state.people.some(p=>p.id===state.selected))state.selected="";
  select.value=state.selected;
  renderEnrollment(); renderProfiles();
}
function renderEnrollment(){
  const person=state.people.find(p=>p.id===state.selected);
  $("enroll-person-info").innerHTML=person?`<span>✓</span><p><strong>${esc(person.name)}</strong><br>${person.sample_count} photos saved · ${person.trained_count} indexed · ${statusLabel(person.status)}</p>`:'<span>◎</span><p>Create a profile to begin capturing.</p>';
  $("sample-counter").textContent=`${person?.sample_count||0} photos`;
  $("enroll-samples").innerHTML=person?.samples.length?person.samples.map(s=>`<div class="sample-item"><img src="${s.image_url}" alt="${esc(person.name)} — ${esc(s.angle)}" loading="lazy"><button class="remove-photo" data-remove-sample="${s.id}" aria-label="Remove ${esc(s.angle)} photo">×</button><small>${esc(s.angle)} ${s.trained?"· built":""}</small></div>`).join(""):'<div class="empty-inline">Your captured photos will appear here.</div>';
  $("train-person").disabled=!person||person.sample_count<3||!!state.job||state.engine!=="ready";
  $("train-person").textContent=person?.status==="ready"?"Rebuild profile":"Build profile";
  updateCapture();
}
function renderProfiles(){
  const query=$("profile-search").value.toLowerCase();
  const people=state.people.filter(p=>p.name.toLowerCase().includes(query));
  $("profiles-grid").innerHTML=people.length?people.map(p=>`<article class="card profile-card"><div class="profile-title"><span class="avatar">${esc(initials(p.name))}</span><div><h2>${esc(p.name)}</h2><small>${p.sample_count} saved photos</small></div><span class="pill ${p.status==="ready"?"":"amber"}">${statusLabel(p.status)}</span></div><div class="profile-images">${p.samples.length?p.samples.slice(0,3).map(s=>`<img src="${s.image_url}" alt="${esc(p.name)} enrollment sample" loading="lazy">`).join(""):'<div class="empty-inline">No photos saved yet</div>'}</div><p class="profile-detail">${p.trained_count} face features indexed<br>${p.sample_count<3?"Add at least 3 clear photos to build this profile.":p.status==="ready"?"Ready for live recognition.":"Build the profile to include your latest photos."}</p><div class="profile-actions"><button class="button secondary" data-edit="${p.id}">View & add photos</button><button class="button dark" data-train="${p.id}" ${p.sample_count<3||state.job||state.engine!=="ready"?"disabled":""}>Build</button><button class="remove-person" data-remove-person="${p.id}" aria-label="Remove ${esc(p.name)}">Remove</button></div></article>`).join(""):`<article class="card empty-inline" style="padding:60px;grid-column:1/-1"><h2>${query?"No matching profiles":"A familiar face starts here."}</h2><p>${query?"Try another name.":"Enroll someone with a few clear photos, or import your existing saved images."}</p><button class="button primary" data-go="enroll">+ Enroll a person</button></article>`;
  $("train-all").disabled=!state.people.some(p=>p.sample_count>=3)||!!state.job||state.engine!=="ready";
  $("import-button").disabled=!!state.job||state.engine!=="ready";
}
async function refresh(){
  if(state.pollBusy)return;
  state.pollBusy=true;
  try{
    const data=await api("/api/state");state.token=data.token;state.engine=data.engine.status;
    $("engine-state").className="pill "+(state.engine==="ready"?"":state.engine==="error"?"red":"amber");
    $("engine-state").innerHTML=`<i></i>${state.engine==="ready"?"Models ready":state.engine==="error"?"Model error":"Loading models"}`;
    $("engine-error").hidden=!data.engine.error;$("engine-error").textContent=data.engine.error||"";
    if(JSON.stringify(state.people)!==JSON.stringify(data.people)){state.people=data.people;renderPeople();}
    else{renderEnrollment();renderProfiles();}
    if(data.job&&!state.job){state.job=data.job.id;watchJob(data.job.id);}
  }catch(error){$("engine-state").className="pill red";$("engine-state").innerHTML="<i></i>Server unavailable";}
  finally{state.pollBusy=false;}
}
function updateCapture(){
  const faces=state.result?.faces||[];
  const valid=!!state.stream&&faces.length===1&&!(faces[0].warnings||[]).length&&state.engine==="ready";
  $("capture-button").disabled=!state.selected||!valid||!!state.job;
  if(state.tab==="enroll"){
    const message=!state.stream?"Start the camera and keep one face in view.":faces.length>1?"More than one face found. Keep just one person in view.":!faces.length?"Looking for a face. Try facing the camera in good light.":(faces[0].warnings||[]).length?faces[0].warnings.join(". "):"Clear face detected. You’re ready to capture.";
    $("capture-feedback").textContent=message;$("capture-feedback").className="capture-feedback "+(valid?"good":state.stream?"bad":"");
  }
}
async function startCamera(){
  if(state.stream)return;
  $("start-camera").disabled=true;
  try{
    const device=$("camera-select").value;
    state.stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:1280},height:{ideal:720},...(device?{deviceId:{exact:device}}:{facingMode:"user"})}});
    const video=$("video");video.srcObject=state.stream;await video.play();
    $("camera-stage").style.aspectRatio=`${video.videoWidth}/${video.videoHeight}`;
    video.hidden=false;$("overlay").hidden=false;$("camera-empty").hidden=true;$("live-indicator").hidden=false;
    $("stop-camera").disabled=false;$("camera-badge").className="pill";$("camera-badge").innerHTML="<i></i>Live camera";
    $("global-camera-stop").hidden=false;
    $("stat-camera").textContent="Connected";$("stat-camera-note").innerHTML='<i class="tiny-dot green"></i> Preview stays on this device';
    const devices=await navigator.mediaDevices.enumerateDevices();$("camera-select").innerHTML=devices.filter(d=>d.kind==="videoinput").map((d,i)=>`<option value="${esc(d.deviceId)}">${esc(d.label||`Camera ${i+1}`)}</option>`).join("");
    $("camera-select").value=state.stream.getVideoTracks()[0].getSettings().deviceId;
    state.generation++;analyzeLoop(state.generation);
  }catch(error){stopCamera();toast(error.name==="NotAllowedError"?"Allow camera access in your browser to start the live view.":`Camera could not start: ${error.message}`,true);}
  finally{$("start-camera").disabled=false;}
}
function stopCamera(){
  state.generation++;state.stream?.getTracks().forEach(t=>t.stop());state.stream=null;state.result=null;
  $("video").srcObject=null;$("video").hidden=true;$("overlay").hidden=true;$("camera-empty").hidden=false;$("live-indicator").hidden=true;
  $("stop-camera").disabled=true;$("camera-badge").className="pill neutral";$("camera-badge").innerHTML="<i></i>Camera off";
  $("global-camera-stop").hidden=true;
  $("stat-camera").textContent="Offline";$("stat-camera-note").innerHTML='<i class="tiny-dot"></i> Ready when you are';
  $("stat-faces").textContent="—";$("stat-posture").textContent="Waiting";$("stat-posture-note").textContent="Start a camera to begin";$("pose-health").textContent="Waiting for camera";
  $("live-people").innerHTML='<div class="empty-small"><span>◎</span><strong>No camera connected</strong><p>Start your camera to see identities here.</p></div>';$("people-count").textContent="0";$("latency").textContent="Analysis —";updateCapture();
  state.session=crypto.randomUUID().replaceAll("-","");
}
function cameraBlob(){
  return new Promise((resolve,reject)=>{
    const video=$("video");if(!video.videoWidth)return reject(new Error("Camera is not ready"));
    const canvas=document.createElement("canvas");const scale=Math.min(1,960/video.videoWidth);
    canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
    canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);canvas.toBlob(blob=>blob?resolve(blob):reject(new Error("Could not read camera frame")),"image/jpeg",.9);
  });
}
async function analyzeLoop(generation){
  if(!state.stream||generation!==state.generation)return;
  const started=performance.now();
  if(state.engine==="ready"&&!state.busy&&["monitor","enroll"].includes(state.tab)&&!document.hidden){
    state.busy=true;
    try{
      const mode=state.tab==="enroll"?"enroll":"monitor";
      const result=await api(`/api/frame?mode=${mode}`,{method:"POST",body:await cameraBlob(),headers:{"Content-Type":"image/jpeg","X-Session-ID":state.session}});
      if(generation===state.generation&&!result.busy){state.result=result;renderLive(result);drawOverlay();updateCapture();}
    }catch(error){if(generation===state.generation){$("latency").textContent=error.message;state.result=null;drawOverlay();updateCapture();}}
    finally{state.busy=false;}
  }
  if(generation===state.generation)setTimeout(()=>analyzeLoop(generation),Math.max(30,220-(performance.now()-started)));
}
function renderLive(result){
  $("stat-faces").textContent=result.faces.length;$("people-count").textContent=result.faces.length;
  $("latency").textContent=`Analysis ${result.processing_ms} ms · live preview independent`;
  const fall=result.fall;const label={NORMAL:"Normal",POSSIBLE_FALL:"Check posture",CONFIRMED_FALL:"Fall event"}[fall.state];
  $("stat-posture").textContent=fall.pose_usable||fall.state!=="NORMAL"?label:"No body pose";
  $("stat-posture").style.color=fall.state==="CONFIRMED_FALL"?"#b74743":fall.state==="POSSIBLE_FALL"?"#9b7021":"";
  $("stat-posture-note").textContent=fall.pose_usable?"Temporal posture analysis active":fall.state!=="NORMAL"?"Pose lost; review the scene":"Keep shoulders and hips in view";
  $("pose-health").textContent=fall.pose_usable?"Visible":"Not visible";
  $("live-people").innerHTML=result.faces.length?result.faces.map(f=>`<div class="person-live"><span class="avatar">${f.person_id?esc(initials(f.name)):"◎"}</span><div><strong>${esc(f.name)}</strong><small>${f.similarity!==null&&f.similarity!==undefined?`Similarity ${f.similarity.toFixed(2)} · `:""}${esc(f.reason||"")}</small></div></div>`).join(""):'<div class="empty-small"><span>◎</span><strong>No face in view</strong><p>Face the camera in good light.</p></div>';
}
function drawOverlay(){
  const canvas=$("overlay"),video=$("video");canvas.width=video.videoWidth||960;canvas.height=video.videoHeight||540;const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!state.result)return;const result=state.result;const sx=canvas.width/result.width,sy=canvas.height/result.height;
  for(const face of result.faces){const [x,y,w,h]=face.bbox;const color=face.person_id?"#b0e9ae":face.name==="Unknown"?"#efd393":"#b8d7e1";ctx.strokeStyle=color;ctx.lineWidth=2.5*sx;ctx.strokeRect(x*sx,y*sy,w*sx,h*sy);const label=face.name;ctx.font=`${Math.round(14*sx)}px Segoe UI`;const tw=ctx.measureText(label).width+20*sx;const ty=Math.max(24*sy,y*sy);ctx.fillStyle="#13372de6";ctx.fillRect(x*sx,ty-25*sy,tw,25*sy);ctx.fillStyle=color;ctx.fillText(label,x*sx+10*sx,ty-8*sy);}
  const points=result.fall.landmarks||[];const lines=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
  ctx.strokeStyle=result.fall.state==="CONFIRMED_FALL"?"#fb9b8c":"#a2d5c3";ctx.lineWidth=2;
  for(const [a,b]of lines){if(!points[a]||!points[b]||points[a][2]<.5||points[b][2]<.5)continue;ctx.beginPath();ctx.moveTo(points[a][0]*canvas.width,points[a][1]*canvas.height);ctx.lineTo(points[b][0]*canvas.width,points[b][1]*canvas.height);ctx.stroke();}
}
async function capturePhoto(){
  $("capture-button").disabled=true;
  try{await api(`/api/people/${state.selected}/samples?angle=${state.angle}`,{method:"POST",body:await cameraBlob(),headers:{"Content-Type":"image/jpeg"}});toast("Photo saved. Try another small angle.");await refresh();}
  catch(error){toast(error.message,true);}finally{updateCapture();}
}
async function startTraining(personId=null){
  try{const job=await api("/api/train",{method:"POST",json:{person_id:personId}});state.job=job.id;renderPeople();watchJob(job.id);}
  catch(error){toast(error.message,true);}
}
async function watchJob(id){
  try{
    const job=await api(`/api/jobs/${id}`);$("job-panel").hidden=false;
    $("job-panel").innerHTML=`<strong>${job.kind==="import"?"Importing saved photos":"Building recognition profiles"}${job.status==="complete"?" · complete":job.status==="error"?" · needs attention":"…"}</strong><br>${job.done} / ${job.total} photos processed · ${job.accepted} accepted${job.status==="running"?`<progress value="${job.done}" max="${Math.max(1,job.total)}"></progress>`:""}${job.errors.length?`<details ${job.status==="running"?"":"open"}><summary>${job.errors.length} item(s) need attention</summary>${job.errors.map(e=>`<div>${esc(e)}</div>`).join("")}</details>`:""}`;
    if(job.status==="running"){setTimeout(()=>watchJob(id),650);return;}
    state.job=null;await refresh();toast(job.status==="error"?job.errors[0]:job.kind==="import"?"Import complete. Review the photos, then build your profiles.":"Face features built. Your ready profiles can now be recognized.",job.status==="error");
  }catch(error){state.job=null;toast(error.message,true);}
}
async function loadEvents(){
  try{const data=await api("/api/events");$("events-body").innerHTML=data.events.length?data.events.map(e=>`<tr><td><span class="pill red">Fall posture</span></td><td>${esc(e.person)}</td><td>${esc(new Date(e.created_at).toLocaleString())}</td><td>Score ${esc(e.details.score)} · ${esc(Object.keys(e.details.signals||{}).join(", "))}</td></tr>`).join(""):'<tr><td colspan="4" class="empty">No events recorded yet. Confirmed posture events will appear here.</td></tr>';}
  catch(error){toast(error.message,true);}
}
let pendingDelete=null;
function askDelete(title,text,action){$("confirm-title").textContent=title;$("confirm-text").textContent=text;pendingDelete=action;$("confirm-dialog").showModal();}
document.addEventListener("click",event=>{
  const button=event.target.closest("button");if(!button)return;
  if(button.dataset.tab)setTab(button.dataset.tab);
  if(button.dataset.go)setTab(button.dataset.go);
  if(button.dataset.angle){state.angle=button.dataset.angle;document.querySelectorAll(".angle").forEach(b=>b.classList.toggle("active",b===button));$("angle-instruction").textContent=angles[state.angle];}
  if(button.dataset.edit){state.selected=button.dataset.edit;renderPeople();setTab("enroll");}
  if(button.dataset.train)startTraining(button.dataset.train);
  if(button.dataset.removePerson){const id=button.dataset.removePerson;const person=state.people.find(p=>p.id===id);askDelete(`Remove ${person.name}?`,"This removes the profile, its saved photos, and recognition features from this workspace.",async()=>{await api(`/api/people/${id}`,{method:"DELETE"});await refresh();toast("Profile removed");});}
  if(button.dataset.removeSample){const id=button.dataset.removeSample;askDelete("Remove this photo?","The saved photo and its recognition features will be removed. Profiles need at least 3 built photos to match.",async()=>{await api(`/api/samples/${id}`,{method:"DELETE"});await refresh();toast("Photo removed");});}
});
$("person-form").addEventListener("submit",async event=>{event.preventDefault();try{const person=await api("/api/people",{method:"POST",json:{name:$("person-name").value}});state.selected=person.id;$("person-name").value="";await refresh();toast("Profile created. Capture a few clear angles.");}catch(error){toast(error.message,true);}});
$("person-select").addEventListener("change",()=>{state.selected=$("person-select").value;renderEnrollment();});
$("profile-search").addEventListener("input",renderProfiles);
$("start-camera").addEventListener("click",startCamera);$("stop-camera").addEventListener("click",stopCamera);
$("global-camera-stop").addEventListener("click",stopCamera);
$("camera-select").addEventListener("change",()=>{if(state.stream){stopCamera();startCamera();}});
$("capture-button").addEventListener("click",capturePhoto);$("train-person").addEventListener("click",()=>startTraining(state.selected));$("train-all").addEventListener("click",()=>startTraining());
$("import-button").addEventListener("click",async()=>{try{const job=await api("/api/import",{method:"POST"});state.job=job.id;renderPeople();watchJob(job.id);}catch(error){toast(error.message,true);}});
$("photo-upload").addEventListener("change",async()=>{
  if(!state.selected){toast("Create or select a profile first",true);$("photo-upload").value="";return;}
  const files=Array.from($("photo-upload").files);let saved=0;const errors=[];
  for(const file of files){try{const data=new FormData();data.append("image",file);data.append("angle","upload");await api(`/api/people/${state.selected}/samples`,{method:"POST",body:data});saved++;}catch(error){errors.push(`${file.name}: ${error.message}`);}}
  $("photo-upload").value="";await refresh();toast(`${saved} photo(s) saved.${errors.length?" "+errors.join(". "):" Build the profile when ready."}`,!!errors.length);
});
$("refresh-events").addEventListener("click",loadEvents);$("cancel-delete").addEventListener("click",()=>{$("confirm-dialog").close();pendingDelete=null;});
$("confirm-delete").addEventListener("click",async()=>{const action=pendingDelete;pendingDelete=null;$("confirm-dialog").close();try{if(action)await action();}catch(error){toast(error.message,true);}});
window.addEventListener("beforeunload",()=>state.stream?.getTracks().forEach(t=>t.stop()));
window.addEventListener("hashchange",()=>{const tab=location.hash.slice(1);if(tab!==state.tab&&titles[tab])setTab(tab);});
$("today").textContent=new Date().toLocaleDateString(undefined,{weekday:"short",day:"numeric",month:"short",year:"numeric"});
setTab(titles[location.hash.slice(1)]?location.hash.slice(1):"monitor");refresh();setInterval(refresh,2000);
