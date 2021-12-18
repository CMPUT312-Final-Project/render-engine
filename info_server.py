# Inspired from https://websockets.readthedocs.io/en/3.0/intro.html
# Actual implementation written from scratch

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

# Main information dictionary
information = {"x":0, "y":0, "z":0, "xr":0, "yr":0}

# Handle server command
async def process_command(websocket, path):
	async for command in websocket:
	    print("Command received: " + command)

	    # Split command parameters
	    params = command.split(" ")
	    # Actual command is always the first word in command sent
	    comm = params[0] # ex. getx

	    if comm[0:3] == "set": # If it is a set command
	    	value = params[1] # Get value
	    	infoName = comm[3:] # Get variable name
	    	information[infoName] = float(value) # Update variable name with value in command
	    	await websocket.send("complete") # Acknowledge command
	    elif comm[0:3] == "get": # If it is a get command
	    	infoName = comm[3:] # Get variable name
	    	value = information[infoName] # Retrieve current variable value
	    	await websocket.send(str(value)) # Sent back the value
	print(information) # Print current information for debugging

# Start a server on local IP (use ipconfig to find this) on port 8765
start_server = websockets.serve(process_command, '172.31.143.89', 8765)

# Constantly try to process commands
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()