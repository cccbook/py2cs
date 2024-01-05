import asyncio
from pyppeteer import launch

target = ''

async def qqmt(source):
    global target
    browser = await launch(headless=False)
    # browser = await launch(headless=True) # 這個延遲要設高一點才抓得到
    page = await browser.newPage()
    await page.goto('https://fanyi.qq.com/')
    await page.type(".textpanel-source .textinput", source)
    await page.waitFor(9000)
    target = await page.evaluate('document.querySelector(".textpanel-target-textblock").innerText')
    await browser.close()
    return target

# source = "Hello , how are you!"
source = 'The number of micro operations is minimized without impacting the quality of the generated code much. For example, instead of generating every possible move between every 32 PowerPC registers, we just generate moves to and from a few temporary registers. These registers T0, T1, T2 are typically stored in host registers by using the GCC static register variable extension.'
print('======== 原文 ========')
print(source)

asyncio.get_event_loop().run_until_complete(qqmt(source))

print('======== 譯文 ========')
print(target)
