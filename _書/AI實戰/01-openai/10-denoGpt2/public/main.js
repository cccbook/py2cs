const menu = document.querySelector("#menu")
const main = document.querySelector('main')

// 頁面：問答
let pageChat = {
    html: `
    <div>
      <div>
        <select id="job" onchange="switchJob()"></select>
        <input type="text" id="question" placeholder="請輸入問題"/>
        <button id="qsubmit" onclick="submitQuestion()">送出問題</button>
      </div>
    </div>
    <div id="questionList">
    </div>
    `,
    init: function () {
        let jobNode = document.querySelector('#job')
        let options = []
        for (let key in jobTemplates) {
            options.push(`<option value="${key}">${key}</option>`)
        }
        jobNode.innerHTML = options.join('\n')
        switchJob()
    }
}

// 頁面：模板
let pageTemplate = {
    html: `
    <select style="visibility: hidden">
      <option value="">　</option>
    </select>
    <textarea id="template">
    </textarea>
    `,
    init: function () {
        let template = document.querySelector('#template')
        template.value = JSON.stringify(jobTemplates, null, 2)
    }
}

let jobTemplates = {
    '模板':'',
    '寫書': '請寫一本主題為 $標題 的書，用 $語言 書寫，章節盡量細分，每章至少要有 5 個小節，章用 第 x 章，小節前面用 1.1, 1.2 這樣的編號，先寫目錄',
    '寫信': '請寫一封主題為 $標題 的信，用 $語言 書寫，要採用 $風格',
    '翻譯': '請將下列文章翻譯成 $目標語言，盡可能翻譯得通順流暢，必要時可以不需要逐句對應',
    '程式翻譯': '請將下列程式轉換成 $目標程式語言 ，函式庫呼叫的部分，保留原來的呼叫名稱，但改用目標語言的語法',
}

const pages = {
    '問答': pageChat,
    '模板': pageTemplate,
}

menu.addEventListener("change", (event) => {
    switchPage()
})

function switchPage() {
    let page = pages[menu.value]
    if (page == null) {
        alert('無此分頁')
        return
    }
    main.innerHTML = page.html
    page.init()
}

switchPage()

function switchJob() {
    let qNode = document.querySelector('#question')
    let jobNode = document.querySelector('#job')
    qNode.value = jobTemplates[jobNode.value]
}

function submitQuestion() {
    let qNode = document.querySelector('#question')
    let question = qNode.value.trim()
    if (question.length == 0) {
        alert('你沒輸入問題，請先輸入後再送出！')
        return
    }
    console.log('start chat')
    // chat(qNode.value).catch((error)=>console.log('error=', error))
    // add a new question
    let newNode = document.createElement('div')
    newNode.innerHTML = `<div> <i class="fa fa-eye"></i> ${question}</div>`
    const qList = document.getElementById("questionList")
    console.log('qList=', qList)
    qList.prepend(newNode)
}

async function chat(question) {
    // let responseNode = document.querySelector('#response')
    let r = await window.fetch(`/chat/${question}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    let response = await r.json()
    console.log(response)
    // responseNode.innerText = response.answer
}


