// Detached JavaScript unit harness: no real browser, location, audio, or network.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync('browser_controls/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = Object.fromEntries(['sound','location','stopLocation','status'].map(id => [id,{}]));
const messages=[], handlers={}, docHandlers={}, timers=[];
let now=0, starts=0, stops=0, geoSuccess, geoFailure;
const context={
  document:{getElementById:id=>nodes[id],addEventListener:(event,fn)=>docHandlers[event]=fn,hidden:false},
  navigator:{geolocation:{watchPosition:(success,failure)=>{geoSuccess=success;geoFailure=failure;return 1;},clearWatch:()=>{}}},
  Date:{now:()=>now},Intl,console,setInterval:fn=>timers.push(fn),
};
context.window={parent:{postMessage:message=>messages.push(message)},addEventListener:(event,fn)=>handlers[event]=fn,
 AudioContext:class {constructor(){this.state='running';this.destination={};}async resume(){}createGain(){return {gain:{},connect(){}};}createOscillator(){return {frequency:{},connect(){},start(){starts++;},stop(){stops++;},disconnect(){}};}}};
vm.createContext(context);vm.runInContext(script,context);
const render=alarm=>handlers.message({source:context.window.parent,data:{type:'streamlit:render',args:{alarm}}});
(async()=>{
 render(true);assert.equal(starts,0); // Permission gesture required.
 await nodes.sound.onclick();render(false);assert.equal(starts,0); // Blink.
 render(true);render(true);assert.equal(starts,1); // No oscillator accumulation.
 render(false);assert.equal(stops,1); // Open stops immediately on command.
 render(true);now=800;timers[0]();assert.equal(stops,2); // Stalled app fails silent.
 nodes.location.onclick();geoFailure({code:1});
 assert.equal(messages.at(-1).value.location,null);assert.equal(messages.at(-1).value.location_status,'denied');
 geoSuccess({coords:{latitude:28.6,longitude:77.2,accuracy:14},timestamp:200000});
 now=2500;geoSuccess({coords:{latitude:28.6,longitude:77.2,accuracy:14},timestamp:200000});
 assert.equal(messages.at(-1).value.location.accuracy,14);
 nodes.stopLocation.onclick();assert.equal(messages.at(-1).value.location,null);
 now=5000;geoSuccess({coords:{latitude:1,longitude:2,accuracy:1},timestamp:300000});assert.equal(messages.at(-1).value.location,null);
 console.log('PASS: browser gesture, blink, alarm start/stop, watchdog, location denial/fix/revocation (mocked).');
})().catch(error=>{console.error(error);process.exitCode=1;});
