# https://websockets.readthedocs.io/en/3.0/intro.html

import asyncio
import websockets

# Commands

# setx <decimal value>
# sety <decimal value>
# setz <decimal value>
# setxr <decimal value>
# setyr <decimal value>

# getx
# gety
# getz
# getxr
# getyr

# Return

# complete

information = {"x":0, "y":0, "z":0, "xr":0, "yr":0}

async def process_command(websocket, path):
	async for command in websocket:
	    print("Command received: " + command)

	    params = command.split(" ")
	    comm = params[0]

	    if comm[0:3] == "set":
	    	value = params[1]
	    	infoName = comm[3:]
	    	#print("Setting " + infoName + " to " + value)
	    	information[infoName] = float(value)
	    	await websocket.send("complete")
	    elif comm[0:3] == "get":
	    	infoName = comm[3:]
	    	value = information[infoName]
	    	#print("Retrieving " + infoName + " (" + str(value) + ")")
	    	await websocket.send(str(value))
	print(information)


start_server = websockets.serve(process_command, '172.31.68.73', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()