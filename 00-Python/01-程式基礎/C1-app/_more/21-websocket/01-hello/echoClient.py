import asyncio
import websockets

async def hello():
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send("Hello world!")
        msg = await websocket.recv()
        print('msg=', msg)

asyncio.run(hello())