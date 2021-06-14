

window.onload = function () {

  let hideEl = document.getElementById("recording")

  const sleep = time => new Promise(resolve => setTimeout(resolve, time));


  URL = window.URL || window.webkitURL;

  var gumStream; 						//stream from getUserMedia()
  var rec; 							//Recorder.js object
  var input;

  // shim for AudioContext when it's not avb. 
  var AudioContext = window.AudioContext || window.webkitAudioContext;
  var audioContext //audio context to help us record

  var recordButton = document.getElementById("record");
  var resetButton = document.querySelector('[data-action="reset"]');


   resetButton.addEventListener('click', () => reset());

  const analysisButton = document.getElementById("analysis_button")

      var wavesurfer = WaveSurfer.create({
      audioContext: audioContext,
      container: '#waveform',
      waveColor: '#D9DCFF',
      progressColor: '#4353FF',
      cursorColor: '#4353FF',
      barWidth: 3,
      barRadius: 3,
      cursorWidth: 1,
      height: 200,
      barGap: 3,
    });



  //add events to those 2 buttons
  var timerInterval = null;


  function startTimer(time) {
     time = time / 1000
    // Update input every second
    return setInterval(function () {

      hideEl.innerHTML = `Recording ends in ${--time} seconds...`;
      if (time < 1) {
//        formattedTime=0
        clear();
      }
    }, 1000);
  }

  const clear = () => {
    clearInterval(timerInterval);
    hideEl.innerHTML = ""


  }


  const handleRec = async () => {

    const buttonPlay = document.getElementById('buttonPlay');

    var recordButton = document.getElementById("record");
    recordButton.style.background ="rgba(15 ,15, 15,0.2)"
    recordButton.setAttribute("disabled","disabled");
    recordButton.style.cursor = "not-allowed";
    buttonPlay.setAttribute("disabled","disabled")
    buttonPlay.classList.add("hide")
    console.log("Was i called")

    startRecording();
    startTimer(15000)
    await sleep(16000)
    stopRecording();


    analysisButton.removeAttribute("disabled")
    analysisButton.classList.remove("hide")
    buttonPlay.removeAttribute("disabled")
    buttonPlay.classList.remove("hide")
    resetButton.removeAttribute("disabled")
    resetButton.classList.remove("hide")

  }



  function startRecording() {
    console.log("recordButton clicked");
    analysisButton.display = "none";
    var constraints = { audio: true, video: false }

    recordButton.disabled = true;

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {

      console.log("getUserMedia() success, stream created, initializing Recorder.js ...");
      audioContext = new AudioContext();
      gumStream = stream;
      input = audioContext.createMediaStreamSource(stream);
      rec = new Recorder(input, { numChannels: 1 })

      rec.record()

      console.log("Recording started");


    }).catch(function (err) {
      recordButton.disabled = false;
      analysisButton.display = block;
    });
  }

  function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    if (rec.recording) {
      //pause
      rec.stop();
      pauseButton.innerHTML = "Resume";
    } else {
      //resume
      rec.record()
      pauseButton.innerHTML = "Pause";

    }
  }

  function stopRecording() {
    console.log("stopButton clicked");
//    recordButton.disabled = false;
    rec.stop();
    gumStream.getAudioTracks()[0].stop();

    var blob = rec.exportWAV(uploadAudio)

  }

  record.addEventListener("click", handleRec);


  const uploadAudio = (blob) => {

    var url = URL.createObjectURL(blob);
    var filename = new Date().toISOString();
    wavesurfer.load(`${url}`);
    const button = document.querySelector('[data-action="play"]');
    buttonPlay.removeAttribute("disabled")
    buttonPlay.classList.remove("hide")

    button.addEventListener('click', wavesurfer.playPause.bind(wavesurfer));

    var fd = new FormData();

    fd.append("record", blob, filename);


    fetch("/audio_dash", {
      method: "POST",
      body: fd
    }).then(res => {
      console.log(res)
      if (res.status === 201) {
        console.log("I responsed")
        console.log(res)
      }
      console.log('Invalid status saving audio message: ' + res.status);


    });

  }

  function reset(){
    wavesurfer.empty()
    recordButton.style.background = "#343b3f"
    recordButton.removeAttribute("disabled");
    recordButton.style.cursor = "default"
    analysisButton.setAttribute("disabled","disabled")
    analysisButton.classList.add("hide")
    buttonPlay.setAttribute("disabled","disabled")
    buttonPlay.classList.add("hide")
    buttonPlay.addEventListener("click",() => wavesurfer.playPause())
    resetButton.classList.add("hide")
    resetButton.setAttribute("disabled","disabled")


  }

  function createDownloadLink(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    var wavesurfer = WaveSurfer.create({
      audioContext: audioContext,
      container: '#waveform',
      waveColor: '#D9DCFF',
      progressColor: '#4353FF',
      cursorColor: '#4353FF',
      barWidth: 3,
      barRadius: 3,
      cursorWidth: 1,
      height: 200,
      barGap: 3
    });
    wavesurfer.load(`${url}`);

    //upload link
    var upload = document.getElementById('stopButton');

    upload.addEventListener("click", function (event) {
      var xhr = new XMLHttpRequest();
      xhr.onload = function (e) {
        if (this.readyState === 4) {
          console.log("Server returned: ", e.target.responseText);
        }
      };
      var fd = new FormData();

      fd.append("record", blob, filename);
      xhr.open("POST", "/audio_dash", true);
      xhr.send(fd);
    })

  }

  analysisButton.addEventListener("click",()=>{
    fetch("/audio_dash",{
      method:"GET",
    }).then(res => {
      console.log(res)
    })
  })


}


