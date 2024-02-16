#!/usr/bin/env python

# WS server example

import asyncio
import websockets

async def hello(websocket, path):
    while True:
        name = await websocket.recv()
        print(f"< {name}")

        greeting = f"Hello {name}!"

        await websocket.send(greeting)
        print(f"> {greeting}")

port = 8765
start_server = websockets.serve(hello, "localhost", port)
print(f'run at http://127.0.0.1:{port}')
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
