const scrollable = document.querySelector('.scrollable'),
    botImg = document.querySelector(".btn"),
    goBtn = document.querySelector(".go-btn"),
    chatWindow = document.querySelectorAll(".window"),
    botIcon = document.querySelector(".bot-button"),
    userTxt = document.querySelector('.user-text'),
    [dots, closeBtn] = document.querySelectorAll(".options"),
    colors = document.querySelector(".colors"),
    colorBtns = document.querySelectorAll(".colors i"),
    sendBtn = document.querySelector(".send-btn")
var bubblestate = false,
    colorOpen = false;

// EVENT LISTENERS
colorBtns.forEach((i) => {
    i.onclick = () => {
        botIcon.style["filter"] = "hue-rotate(" + i.getAttribute("hue-val") + "deg)"
        colorBtns.forEach((j) => {
            j.classList.remove("active")
            j.style["filter"] = "hue-rotate(-" + i.getAttribute("hue-val") + "deg)"
        })
        i.classList.add("active")
        setTimeout(() => {
            colors.classList.remove("overflowoverride")
            colors.classList.toggle("shrink")
            colorOpen = false
        }, 800);
    }
})

botImg.onclick = closeBtn.onclick = () => {
    botIcon.classList.toggle("expand")
    colors.classList.add("shrink")
    colors.classList.remove("overflowoverride")
    colorOpen = false
}

dots.onclick = () => {
    colors.classList.toggle("shrink")
    if (!colorOpen) {
        setTimeout(() => {
            colors.classList.toggle("overflowoverride")
            colorOpen = true;
        }, 500);
    } else {
        colors.classList.toggle("overflowoverride")
        colorOpen = false
    }

}

goBtn.addEventListener("click", () => {
    scrollToEnd(scrollable)
    chatWindow.forEach((i) => {
        i.classList.add("move-out")
    })
    setTimeout(() => {
        document.querySelector(".topbar2").remove()
    }, 1500);
})

userTxt.addEventListener("input", (e) => {
    if (!bubblestate) {
        scrollable.insertAdjacentHTML('beforeend', `<div class="question-wrap">
        <div class="question load animate__animated animate__fadeInUp animate__fast">
            <span></span>
            <span></span>
            <span></span>
            </div>
    </div>`)
        bubblestate = true;
    }
    if (userTxt.value == "") {
        x = [...document.querySelectorAll(".question.load")].pop()
        x.classList.add("animate__fadeOutRight")
        setTimeout(() => {
            scrollToEnd(scrollable, scrollable.getBoundingClientRect().height + x.parentNode.getBoundingClientRect().height)
        }, 400);
        setTimeout(() => {
            x.parentNode.remove()
        }, 600);
        bubblestate = false;
    }
    scrollToEnd(scrollable, 80)

})
userTxt.addEventListener('keydown', function (event) {
    const key = event.key;
    if (key == "Enter" && userTxt.value!="") {
        console.log(key == "Enter")
        x = [...document.querySelectorAll(".question.load")].pop()
        x.parentNode.remove()
        x.remove()
        sendBtn.click()
        scrollToEnd(scrollable)
        bubblestate = false;
    }
});
sendBtn.addEventListener("click", () => {
    if (userTxt.value != ""){
        fetch('/chatbot', {
            method: 'POST',
            headers: {
                'Accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "query": userTxt.value
            })
        }).then(res => res.json())
        .then(res => {
            if(res["response"]=="SERVER ERROR"){
                setTimeout(() => {
                    x = document.querySelector(".response.load")
                    x.classList.add("animate__fadeOutDown")
                    console.log(x)
                    x.parentNode.remove()
                    scrollable.insertAdjacentHTML('beforeend', `
        <div class="response-wrap">
        <div class="response error animate__animated animate__fadeInUp animate__faster">
            ${res["response"]}<p class="time">${h}:${m}</p>
        </div>
    </div>`)
                scrollToEnd(scrollable)
                }, 600);
            }
            else{
                x = document.querySelector(".response.load")
                x.classList.add("animate__fadeOutDown")
                console.log(x)
                x.parentNode.remove()
                scrollable.insertAdjacentHTML('beforeend', `
        <div class="response-wrap">
        <div class="response animate__animated animate__fadeInUp animate__faster">
            ${res["response"]}<p class="time">${h}:${m}</p>
        </div>
    </div>`)
                scrollToEnd(scrollable)
            }
        });
    console.log("Fired")
    var x = new Date()
    m = x.getMinutes();
    h = x.getHours();
    console.log(h, m)
    h = h < 10 ? "0" + h : h;
    m = m < 10 ? "0" + m : m;
    setTimeout(() => {
        scrollable.insertAdjacentHTML('beforeend', `<div class="response-wrap">
  <div class="response load animate__animated animate__fadeInUp animate__fast">
      <span></span>
      <span></span>
      <span></span>
      </div>
</div>`)
        scrollToEnd(scrollable)
    }, 600);
    scrollable.insertAdjacentHTML('beforeend', `
    <div class="question-wrap">
    <div class="question animate__animated animate__fadeInUp animate__faster">
        ${userTxt.value}<p class="time">${h}:${m}</p>
    </div>
</div>`)
    userTxt.value = ""
}
})

function scrollToEnd(x, y = 0) {
    x.scrollTo(0, x.scrollHeight - y)
}