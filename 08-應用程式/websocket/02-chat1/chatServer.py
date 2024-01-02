import asyncio
import websockets

async def reply(websocket):
    async for message in websocket:
        print('receive:'+message)
        await websocket.send(message+' received!')

async def main():
    async with websockets.serve(reply, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())