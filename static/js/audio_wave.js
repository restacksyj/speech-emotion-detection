

window.onload = function () {
    let hideEl = document.getElementById("recording")

    let mainbtn = document.getElementById("mainbtn")

    mainbtn.addEventListener("click", () => {
        hideEl.innerHTML = "Recording..."
        
        setInterval(() => {
            hideEl.innerHTML = ""
            mainbtn.disabled = false
        }, 16000)


    })

}


