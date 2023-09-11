import { Application, Router, send } from "https://deno.land/x/oak/mod.ts";
import * as lib from './lib.js'

const app = new Application()
const router = new Router()

router.get('/', (ctx)=>ctx.response.redirect('/public/index.html'))
  .get('/public/(.*)', pub)
  .get('/chat/:question', chat)

app.use(router.routes())
app.use(router.allowedMethods())

async function pub(ctx) {
  console.log('path=', ctx.request.url.pathname)
  await send(ctx, ctx.request.url.pathname, {
    root: `${Deno.cwd()}/`,
    index: "index.html",
  })
}

async function chat (ctx) {
    const q = ctx.params.question
    let r = await lib.chat(q)
    ctx.response.type = 'application/json'
    ctx.response.body = r
}
  
/*
async function list (ctx) {
  ctx.response.type = 'application/json'
  ctx.response.body = posts
}

async function show (ctx) {
  const id = ctx.params.id
  const post = posts[id]
  if (!post) ctx.throw(404, 'invalid post id')
  ctx.response.type = 'application/json'
  ctx.response.body = post
}

async function create (ctx) {
  // var post = ctx.request.body
  const body = ctx.request.body(); // content type automatically detected
  console.log('body = ', body)
  if (body.type === "json") {
    let post = await body.value
    post.id = posts.length
    posts.push(post)
    ctx.response.body = 'success'
    console.log('create:save=>', post)
  }
}
*/
console.log('Server run at http://127.0.0.1:8001')
await app.listen({ port: 8001 })
