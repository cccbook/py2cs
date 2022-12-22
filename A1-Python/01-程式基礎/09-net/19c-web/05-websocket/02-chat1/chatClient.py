import asyncio
import websockets

async def chat():
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            msg = input("? ")
            if msg == 'exit': break
            await websocket.send(msg)
            reply = await websocket.recv()
            print(f'reply:{reply}')

asyncio.run(chat())