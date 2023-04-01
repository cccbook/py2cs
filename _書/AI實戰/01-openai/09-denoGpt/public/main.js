let jobTemplate = {
    '寫書':'請寫一本主題為 $標題 的書，用 $語言 書寫，章節盡量細分，每章至少要有 5 個小節，章用 第 x 章，小節前面用 1.1, 1.2 這樣的編號，先寫目錄',
    '寫信':'請寫一封主題為 $標題 的信，用 $語言 書寫，要採用 $風格 風格',
    '翻譯':'請將下列文章翻譯成 $目標語言，盡可能翻譯得通順流暢，必要時可以不需要逐句對應',
    '程式翻譯':'請將下列程式轉換成 $目標程式語言 ，函式庫呼叫的部分，保留原來的呼叫名稱，但改用目標語言的語法',
}
const menu = document.querySelector("#menu")
const main = document.querySelector('main')

const pages = {
'開始使用':`
<select id="job" onchange="switchJob()">
  <option value="寫書">寫書</option>
  <option value="翻譯">翻譯</option>
</select>

<textarea id="question" placeholder="問題">
</textarea>

<textarea id="source" placeholder="輸入文章">
</textarea>

<div id="target" placeholder="輸出文章">
</div>
`,
'編輯模板':`
<select>
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
    switchJob()
}

function switchJob() {
    let question = document.querySelector('#question')
    let job = document.querySelector('#job')
    question.value = jobTemplate[job.value]
}

// 編輯模板

function pageEditInit() {
    let template = document.querySelector('#template')
    template.value = JSON.stringify(jobTemplate, null, 2)
}
