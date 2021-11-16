# https://websockets.readthedocs.io/en/3.0/intro.html

import asyncio
import websockets

async def getPositionAndRotation():
	infoNeeded = ['x','y','z']
	values = []
	for name in infoNeeded:
	    async with websockets.connect('ws://172.31.73.76:8765') as websocket:
	        await websocket.send("get"+name)
	        value = await websocket.recv()
	        values.append(float(value))

asyncio.get_event_loop().run_until_complete(getPositionAndRotation())