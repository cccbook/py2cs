import asyncio
import websockets

clients = []

async def reply(websocket):
    clients.append(websocket)
    async for message in websocket:
        print('receive:'+message)
        for client in clients: # broadcast
            try:
                await client.send(message)
            except:
                pass

async def main():
    async with websockets.serve(reply, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())