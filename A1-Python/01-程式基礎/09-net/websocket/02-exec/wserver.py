#!/usr/bin/env python

# WS server example
import json
import asyncio
import websockets
import sys
import io

def run(code):
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    exec(code)
    mystdout.flush()
    out = mystdout.getvalue()
    mystdout.close()
    sys.stdout = old_stdout
    return out

async def main(websocket, path):
    while True:
        jsonText = await websocket.recv()
        print(f"{jsonText}")
        jsonObj = json.loads(jsonText)
        out = run(jsonObj['code'])
        jsonObj['out'] = out
        response = json.dumps(jsonObj)
        await websocket.send(response)

port = 8765
start_server = websockets.serve(main, "localhost", port)
print(f'run at http://127.0.0.1:{port}')
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
