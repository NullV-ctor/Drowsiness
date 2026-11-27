// Mock camera, clock and component messages; no device or network is accessed.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('browser_controls/index.html','utf8');
const nodes=Object.fromEntries(['sound','location','stopLocation','status','camera','preview','cameraStatus','startCamera','stopCamera'].map(id=>[id,{style:{}}]));
let now=10000,stopped=0,denied=false;
Object.assign(nodes.preview,{readyState:2,videoWidth:640,videoHeight:480,play:async()=>{}});
const handlers={},docHandlers={},timers=[],messages=[];
const context={document:{body:{scrollHeight:620},getElementById:id=>nodes[id],createElement:()=>({getContext:()=>({drawImage(){}}),toDataURL:()=> 'data:image/jpeg;base64,fixture'}),addEventListener:(k,f)=>{(docHandlers[k]??=[]).push(f)},hidden:false},
 navigator:{mediaDevices:{getUserMedia:async()=>{if(denied)throw {name:'NotAllowedError'};return {getTracks:()=>[{stop(){stopped++}}]}}}},
 Date:{now:()=>now},Intl,console,setInterval:fn=>timers.push(fn)};
context.window={parent:{postMessage:m=>messages.push(m)},addEventListener:(k,f)=>{(handlers[k]??=[]).push(f)}};
vm.createContext(context);
for(const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(match[1],context);
const render=(ack=0)=>handlers.message.forEach(f=>f({source:context.window.parent,data:{type:'streamlit:render',args:{camera:true,ack,alarm:false}}}));
const capture=()=>timers[1]();
(async()=>{
 render();await nodes.startCamera.onclick();capture();
 const first=messages.at(-1).value;assert.ok(first.frame);const count=messages.length;
 capture();assert.equal(messages.length,count,'Only one frame may be in flight');
 render(first.frame_id);now+=200;capture();assert.notEqual(messages.at(-1).value.frame_id,first.frame_id);
 now+=4000;capture();assert.match(nodes.cameraStatus.textContent,/slow/);
 nodes.stopCamera.onclick();assert.equal(stopped,1);assert.equal(messages.at(-1).value.camera_active,false);
 denied=true;await nodes.startCamera.onclick();assert.match(nodes.cameraStatus.textContent,/permission denied/);
 denied=false;await nodes.startCamera.onclick();context.document.hidden=true;docHandlers.visibilitychange.forEach(f=>f());assert.equal(messages.at(-1).value.camera_active,false);
 context.document.hidden=false;
 let resolvePermission;
 context.navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{resolvePermission=resolve;});
 const request=nodes.startCamera.onclick();
 assert.equal(nodes.startCamera.disabled,true);assert.equal(nodes.stopCamera.disabled,false);
 nodes.stopCamera.onclick();const stoppedBefore=stopped;
 resolvePermission({getTracks:()=>[{stop(){stopped++}}]});await request;
 assert.equal(stopped,stoppedBefore+1,'A late permission response must release the camera after Stop');
 assert.equal(messages.at(-1).value.camera_active,false);
 assert.equal(nodes.startCamera.disabled,false);assert.equal(nodes.stopCamera.disabled,true);
 console.log('PASS HTTPS camera: one frame in flight, acknowledgement, slow response, stop, permission denial, hidden tab, canceled start and late permission response.');
})().catch(e=>{console.error(e);process.exitCode=1});
