let jobTemplates = {
    '寫書':'請寫一本主題為 $標題 的書，用 $語言 書寫，章節盡量細分，每章至少要有 5 個小節，章用 第 x 章，小節前面用 1.1, 1.2 這樣的編號，先寫目錄',
    '寫信':'請寫一封主題為 $標題 的信，用 $語言 書寫，要採用 $風格',
    '翻譯':'請將下列文章翻譯成 $目標語言，盡可能翻譯得通順流暢，必要時可以不需要逐句對應',
    '程式翻譯':'請將下列程式轉換成 $目標程式語言 ，函式庫呼叫的部分，保留原來的呼叫名稱，但改用目標語言的語法',
}
const menu = document.querySelector("#menu")
const main = document.querySelector('main')

const pages = {
'開始使用':`
<select id="job" onchange="switchJob()">

</select>

<textarea id="question" placeholder="問題">
</textarea>

<button onclick="chat()">送出問題</button>

<textarea id="input" placeholder="輸入文章">
</textarea>

<div id="response" placeholder="輸出文章">
</div>
`,
'編輯模板':`
<select style="visibility: hidden">
  <option value="">　</option>
</select>
<textarea id="template">
</textarea>
`,
}

let menuScript = {
    '開始使用':pageUseInit,
    '編輯模板':pageEditInit,
}

menu.addEventListener("change", (event) => {
    switchPage()
})

function switchPage() {
    main.innerHTML = pages[menu.value]
    menuScript[menu.value]()
}

switchPage()

// 開始使用
function pageUseInit() {
    let jobNode = document.querySelector('#job')
    let options = []
    for (let key in jobTemplates) {
        options.push(`<option value="${key}">${key}</option>`)
    }
    jobNode.innerHTML = options.join('\n') 
    switchJob()
}

function switchJob() {
    let qNode = document.querySelector('#question')
    let jobNode = document.querySelector('#job')
    qNode.value = jobTemplates[jobNode.value]
}

async function chat() {
    let qNode = document.querySelector('#question')
    let responseNode = document.querySelector('#response')
    let r = await window.fetch(`/chat/${qNode.value}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
      })
    let response = await r.json()
    responseNode.innerText = response.answer
}
// 編輯模板

function pageEditInit() {
    let template = document.querySelector('#template')
    template.value = JSON.stringify(jobTemplates, null, 2)
}
