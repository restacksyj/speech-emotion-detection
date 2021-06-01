

window.onload = function () {

  let hideEl = document.getElementById("recording")

  const sleep = time => new Promise(resolve => setTimeout(resolve, time));

  // let mainbtn = document.getElementById("mainbtn")

  // mainbtn.addEventListener("click", () => {
  //     hideEl.innerHTML = "Recording..."

  //     setInterval(() => {
  //         hideEl.innerHTML = ""
  //         mainbtn.disabled = false
  //     }, 16000)


  // })

  URL = window.URL || window.webkitURL;

  var gumStream; 						//stream from getUserMedia()
  var rec; 							//Recorder.js object
  var input; 							//MediaStreamAudioSourceNode we'll be recording

  // shim for AudioContext when it's not avb. 
  var AudioContext = window.AudioContext || window.webkitAudioContext;
  var audioContext //audio context to help us record

  var recordButton = document.getElementById("record");
  const analysisButton = document.getElementById("analysis_button")


  // var stopButton = document.getElementById("stopButton");
  // var pauseButton = document.getElementById("pauseButton");

  //add events to those 2 buttons
  var timerInterval = null;

  function startTimer(time) {
    formattedTime = time / 1000
    // Update input every second
    return setInterval(function () {
      
      hideEl.innerHTML = `Recording ends in ${--formattedTime} seconds...`;
      if (formattedTime < 1) {
        clear();
      }
    }, 1000);
  }

  const clear = () => {
    clearInterval(timerInterval);
    hideEl.innerHTML = ""
  }


  const handleRec = async () => {

    var recordButton = document.getElementById("record");
    recordButton.style.background ="rgba(15 ,15, 15,0.2)"
    recordButton.setAttribute("disabled","disabled");

    startRecording();
    startTimer(15000)
    await sleep(16000)
    stopRecording();
    recordButton.style.background = "#343b3f"
    recordButton.removeAttribute("disabled");
    analysisButton.removeAttribute("disabled")
    analysisButton.classList.remove("hide")

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

    // //disable the stop button, enable the record too allow for new recordings
    // stopButton.disabled = true;
    recordButton.disabled = false;
    // pauseButton.disabled = true;

    // //reset button just in case the recording is stopped while paused
    // pauseButton.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();


    // var wavesurfer = WaveSurfer.create({
    //   audioContext: audioContext,
    //   container: '#waveform',
    //   waveColor: '#D9DCFF',
    //   progressColor: '#4353FF',
    //   cursorColor: '#4353FF',
    //   barWidth: 3,
    //   barRadius: 3,
    //   cursorWidth: 1,
    //   height: 200,
    //   barGap: 3
    // });
    // wavesurfer.load(`${url}`);

    //create the wav blob and pass it on to createDownloadLink
    var blob = rec.exportWAV(uploadAudio)



    //rec.exportWAV(createDownloadLink);
    // var xhr = new XMLHttpRequest();
    // xhr.onload = function (e) {
    //   if (this.readyState === 4) {
    //     console.log("Server returned: ", e.target.responseText);
    //   }
    // };
    // var fd = new FormData();

    // fd.append("record", blob, filename);
    // xhr.open("POST", "/audio_dash", true);
    // xhr.send(fd);
  }

  record.addEventListener("click", handleRec);


  const uploadAudio = (blob) => {
    var url = URL.createObjectURL(blob);
    var filename = new Date().toISOString();


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
    wavesurfer.load(`${url}`);
    const button = document.querySelector('[data-action="play"]');
    button.removeAttribute("disabled")
    button.classList.remove("hide")

    button.addEventListener('click', wavesurfer.playPause.bind(wavesurfer));

    // var xhr = new XMLHttpRequest();
    // xhr.onload = function (e) {
    //   if (this.readyState === 4) {
    //     console.log("Server returned: ", e.target.responseText);
    //   }
    // };
    var fd = new FormData();

    fd.append("record", blob, filename);
    // xhr.open("POST", "/audio_dash", true);
    // xhr.send(fd);

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


