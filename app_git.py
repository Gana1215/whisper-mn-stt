rec = components.html(
    """
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
      <button id="recbtn" style="
          background:#0f4c81;color:white;border:none;border-radius:50%;
          width:110px;height:110px;font-size:1.1rem;box-shadow:0 4px 12px rgba(0,0,0,0.3);
          transition:all 0.25s ease;">🎙️ Start</button>
      <div id="status" style="color:#555;font-size:0.95rem;">Tap Start, speak, then tap Stop</div>
    </div>

    <script>
    let rec, chunks=[];
    const btn=document.getElementById("recbtn");
    const status=document.getElementById("status");

    // Initial handshake to Streamlit (prevents 'Bad message format')
    window.parent.postMessage({type:'streamlit:setComponentValue',value:""}, '*');

    async function startRec(){
      try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        rec=new MediaRecorder(stream);
        chunks=[];
        rec.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data)};
        rec.onstop=async()=>{
          const blob=new Blob(chunks,{type:'audio/webm'});
          status.textContent="Processing...";
          const arrayBuffer=await blob.arrayBuffer();
          const bytes=Array.from(new Uint8Array(arrayBuffer));
          const json=JSON.stringify({audio:bytes,mime:blob.type||'audio/webm'});
          // Dual send to ensure reception
          window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*');
          setTimeout(()=>window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*'),50);
          status.textContent="✅ Audio sent";
        };
        rec.start();
        btn.textContent="⏹ Stop";
        btn.style.background="#d32f2f";
        status.textContent="Recording...";
      }catch(err){
        status.textContent="❌ Mic permission denied";
      }
    }

    function stopRec(){
      if(rec && rec.state!=='inactive'){
        rec.stop();
        btn.textContent="🎙️ Start";
        btn.style.background="#0f4c81";
        status.textContent="Stopping...";
      }
    }

    btn.addEventListener('click',()=>{
      if(!rec || rec.state==='inactive') startRec(); else stopRec();
    });
    </script>
    """,
    height=230,
    scrolling=False,
    key="recorder_v432"
)
