# https://codeloop.org/python-modern-opengl-perspective-projection/

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from PIL import Image
import math

import asyncio
import websockets


# Main global positional values
values = [0,0,0,0,0] # Values obtained from the information server
offsets = [-0.5, -18, 10, 0.01, 0] # Offsets of values manually calibrated
offsets = [-1.0, -18, 4, -0.11, 0]
	# FORMAT = [x_position, y_position, z_position, x_rotation, y_rotation]

server_address = 'ws://172.31.143.89:8765' # Information server address

"""
	getPositionAndRotation Function
		sends get requests to the information server to get current position information
"""
async def getPositionAndRotation():
	infoNeeded = ['x','y','z','xr','yr'] # Information server value tags
	for i in range(len(infoNeeded)):
		name = infoNeeded[i]
		async with websockets.connect(server_address) as websocket: # Each value is individually requested
			await websocket.send("get"+name) # Send get command to server, ex: "getx"
			value = await websocket.recv()
			values[i] = float(value) # Sets the values array to response information from server

"""
	inputEvent Function
		processes any input event in the GLFW window (active application window framework)
"""
def inputEvent(window, key, scancode, action, mods):
	increment = 0.5
	z_increment = 2
	if action == glfw.PRESS: # On keypress modify offsets to manually calibrate setup
		if key == glfw.KEY_W:
			offsets[1] += increment
		elif key == glfw.KEY_S:
			offsets[1] -= increment
		elif key == glfw.KEY_A:
			offsets[0] += increment
		elif key == glfw.KEY_D:
			offsets[0] -= increment
		elif key == glfw.KEY_Q:
			offsets[2] += z_increment
		elif key == glfw.KEY_E:
			offsets[2] -= z_increment
		elif key == glfw.KEY_UP:
			offsets[3] += 0.01
		elif key == glfw.KEY_DOWN:
			offsets[3] -= 0.01
		elif key == glfw.KEY_LEFT:
			offsets[4] += 0.01
		elif key == glfw.KEY_RIGHT:
			offsets[4] -= 0.01
		elif key == glfw.KEY_ESCAPE: # Creature comfort, escape safely terminates the program
			glfw.set_window_should_close(window, GL_TRUE)

"""
	createTransformationMatrix Function
		given x position, y position, z position, x rotation and y rotation
		returns a 4x4 transformation matrix from given translation and rotation
"""
def createTransformationMatrix(xPos=0, yPos=0, zPos=0, xRotation=0, yRotation=0, zRotation=0):
	# Individual rotation matricies for x and y rotation
	rot_x = pyrr.Matrix44.from_x_rotation(xRotation)
	rot_y = pyrr.Matrix44.from_y_rotation(yRotation)

	transformationMatrix = rot_x @ rot_y # Combine rotation matricies

	transformationMatrix[3][0] = xPos # Set translation for x, y and z
	transformationMatrix[3][1] = yPos
	transformationMatrix[3][2] = zPos

	return transformationMatrix

"""
	main Function
		starts application and handles major functionality
"""
def main():
	if not glfw.init(): # If GLFW fails to initialize, terminate program
		return
 
	display = [1920, 1080] # Target display resolution

	monitors = glfw.get_monitors() # Get available monitors and select for secondary monitor (projector display)
	active_monitor = monitors[0]

	# GLFW call to create the application window
	window = glfw.create_window(display[0], display[1], "Pyopengl Perspective Projection", active_monitor, None)
 
	if not window: # If window failed to create, terminate program
		glfw.terminate()
		return
 
	glfw.make_context_current(window) # Set GLFW context to selected window

	# 3D MODEL DATA AND SHADERS #

	# cube width from origin (each side is 3.5 cm from the origin of the 7x7cm cube)
	half_cube_width = 7/2

	# Single texture model (single texture on every side)
	# Each vertice follows the following pattern:
	#		x pos., y pos., z pos., default color red, blue, green, x texture coordinate, y texture coordinate
	# NOTE: color is unused but causes memory pointer errors if removed from model data
	single_tex_cube = [-half_cube_width, -half_cube_width, half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			half_cube_width, -half_cube_width, half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			-half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			half_cube_width, half_cube_width, -half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			-half_cube_width, half_cube_width, -half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			half_cube_width, half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			half_cube_width, -half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-half_cube_width, half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			-half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			-half_cube_width, -half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			-half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			half_cube_width, -half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			-half_cube_width, -half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			half_cube_width, half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 0.0, 0.0,
			-half_cube_width, half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 1.0, 0.0,
			-half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 1.0, 1.0,
			half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 0.0, 1.0]

	# Multi texture model (unique texture on each side)
	tex_width = 384 # Reference pixel values to simplify texture coordinate calculations
	tex_height = 512
	# Same configuration as the single texture model except texture coordinates in
	# order to divide reference texture into individual textures for each side (see grass.jpg)
	multi_tex_cube = [-half_cube_width, -half_cube_width, half_cube_width, 1.0, 0.0, 0.0, 	128.0/tex_width, 128.0/tex_height,
			half_cube_width, -half_cube_width, half_cube_width, 0.0, 1.0, 0.0, 				256.0/tex_width, 128.0/tex_height,
			half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 				256.0/tex_width, 0.0/tex_height,
			-half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 				128.0/tex_width, 0.0/tex_height,
 
			-half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 			128.0/tex_width, 256.0/tex_height,
			half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 			256.0/tex_width, 256.0/tex_height,
			half_cube_width, half_cube_width, -half_cube_width, 0.0, 0.0, 1.0, 				256.0/tex_width, 384.0/tex_height,
			-half_cube_width, half_cube_width, -half_cube_width, 1.0, 1.0, 1.0, 			128.0/tex_width, 384.0/tex_height,
 
			half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 			256.0/tex_width, 256.0/tex_height,
			half_cube_width, half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 				384.0/tex_width, 256.0/tex_height,
			half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 				384.0/tex_width, 128.0/tex_height,
			half_cube_width, -half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 				256.0/tex_width, 128.0/tex_height,
 
			-half_cube_width, half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 			0.0/tex_width, 256.0/tex_height,
			-half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 			128.0/tex_width, 256.0/tex_height,
			-half_cube_width, -half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 			128.0/tex_width, 128.0/tex_height,
			-half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 				0.0/tex_width, 128.0/tex_height,
 
			-half_cube_width, -half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 			256.0/tex_width, 128.0/tex_height,
			half_cube_width, -half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 			128.0/tex_width, 128.0/tex_height,
			half_cube_width, -half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 				128.0/tex_width, 256.0/tex_height,
			-half_cube_width, -half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 			256.0/tex_width, 256.0/tex_height,
 
			half_cube_width, half_cube_width, -half_cube_width, 1.0, 0.0, 0.0, 				128.0/tex_width, 512.0/tex_height,
			-half_cube_width, half_cube_width, -half_cube_width, 0.0, 1.0, 0.0, 			256.0/tex_width, 512.0/tex_height,
			-half_cube_width, half_cube_width, half_cube_width, 0.0, 0.0, 1.0, 				256.0/tex_width, 384.0/tex_height,
			half_cube_width, half_cube_width, half_cube_width, 1.0, 1.0, 1.0, 				128.0/tex_width, 384.0/tex_height]

	# Move model into numpy friendly array
	active_model = multi_tex_cube
	#active_model = single_tex_cube
	cube = np.array(active_model, dtype=np.float32)
 
 	# Cube model indicies for conversion to polygons
	indices = [0, 1, 2, 2, 3, 0, # ex: front face is composed of 2 triangles that share verticies 0 and 2
			   4, 5, 6, 6, 7, 4,
			   8, 9, 10, 10, 11, 8,
			   12, 13, 14, 14, 15, 12,
			   16, 17, 18, 18, 19, 16,
			   20, 21, 22, 22, 23, 20]
 
 	# Convert indicies into numpy friendly array
	indices = np.array(indices, dtype=np.uint32)

	# SHADERS - used from tutorial (https://codeloop.org/python-modern-opengl-perspective-projection/)
	# 	Lightly modified

 	# Vertex Shader for cube rendering
 	# Simply takes in position, texture coordinates, the transformation matrix for the cube
 	# and the view, model, and projection matricies
 	# returns the gl_position of each pixel to be rendered to the screen
 	# also passes newColor and texture coordinates to fragment shader
	VERTEX_SHADER = """
 
			  #version 330
 
			  in vec3 position;
			  in vec2 InTexCoords;
 
			  out vec2 OutTexCoords;
 
			  uniform mat4 transform; 
			  
			  
			  uniform mat4 view;
			  uniform mat4 model;
			  uniform mat4 projection;
 
			  void main() {
 
				gl_Position = projection * view * model * transform * vec4(position, 1.0f);
			   OutTexCoords = InTexCoords;
 
				}
 
 
		  """
 	
 	# Fragment Shader for setting specific color of pixel
 	# Simply sets the output color for the pixel from the texture coordinates and sampler image
	FRAGMENT_SHADER = """
		   #version 330
 
			in vec2 OutTexCoords;
 
			out vec4 outColor;
			uniform sampler2D samplerTex;
 
		   void main() {
 
			  outColor = texture(samplerTex, OutTexCoords);
 
		   }
 
	   """
 
	# Compile The Program and shaders
 
 	# Compile cube shader in order for OpenGL to process
	shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
											  OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
 
 	# Bind model VBO (variable buffer object) into memory
	VBO = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, VBO)
	glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)
 
 
	# Bind index EBU (extra buffer object) into memory
	EBO = glGenBuffers(1)
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)
 
 
	# Bind position data from model to shader
	position = glGetAttribLocation(shader, 'position')
	glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(0))
	glEnableVertexAttribArray(position)
 
 	# Bind texture coordinate data from model to shader
	texCoords = glGetAttribLocation(shader, "InTexCoords")
	glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(24))
	glEnableVertexAttribArray(texCoords)
 
 	# Generate a texture object and bind it to shader
	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
 
 
 	# TEXTURE APPEARANCE PARAMETERS

	# Set the texture wrapping parameters (texture coords will loop back if beyond image domain)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
 
	# Set texture filtering parameters (linear filtering, so zero antialiasing on texture)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
 
 
	# load images into memory from local files

	# Loads Rick Astley animation
	images_to_load = []
	for i in range(26):
		images_to_load.append("rick_anim/frame_"+str(i)+".jpg")

	# Loads Minecraft Grass block multisided image
	images_to_load = ["grass.jpg"]

	images = []
	imgs_data = []

	# For every predefined image, load it into memory and keep data for OpenGL handling
	for im in images_to_load:
		image = Image.open(im)
		images.append(image)
		imgs_data.append(np.array(list(image.getdata()), np.uint8))

	# Bind image to texture for use in shader
	img_id = 0
	image = images[img_id]
	img_data = imgs_data[img_id]
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
	glEnable(GL_TEXTURE_2D)
 
 	# OpenGL call to bind shader to OpenGL pipeline
	glUseProgram(shader)
 
 	# Background color and depth based rendering (nearest textures rendered on top)
	glClearColor(0.0, 0.0, 0.0, 1.0)
	glEnable(GL_DEPTH_TEST)
 
 
	# Creating projection, view and model matricies
	view = pyrr.matrix44.create_from_translation(pyrr.Vector3([1.0,6.0,-134.0])) # Projector global offset
	view = view @ pyrr.Matrix44.from_x_rotation((6.56/180*math.pi)) # Projector global rotation offset applied (6.56 degrees)
	projection = pyrr.matrix44.create_perspective_projection(20, display[0]/display[1], 0.1, 1000.0) # Sets projection matrix with 20 degree FOV
	model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0,0.0,0.0])) # Global origin

	# Bind matricies to shader
	view_loc = glGetUniformLocation(shader, "view")
	proj_loc = glGetUniformLocation(shader, "projection")
	model_loc = glGetUniformLocation(shader, "model")

	glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

	# Bind input callback function
	glfw.set_key_callback(window, inputEvent)
 
 	# Application loop
	while not glfw.window_should_close(window):

		# Select image from loaded images and rebind in shader, switching at 5 FPS
		img_id = int(glfw.get_time()/(1/5))%len(images_to_load)
		image = images[img_id]
		img_data = imgs_data[img_id]
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
		glEnable(GL_TEXTURE_2D)

		# Poll events to trigger any input events
		glfw.poll_events()
 
 		# Clear screen for new frame to render
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
 
 		# Update positional information from the information server
		asyncio.get_event_loop().run_until_complete(getPositionAndRotation())

		# Print offsets for clarity in manual calibration
		print(offsets)

		# Set transformation matrix of the cube and bind to shader
		transformation = createTransformationMatrix(-values[0]+offsets[0], values[1]+offsets[1], -values[2]+offsets[2], -values[3]+offsets[3], -values[4]+offsets[4], 0.0)
		# transformation = createTransformationMatrix(-values[0]+offsets[0], values[1]+offsets[1], -values[2]+offsets[2], glfw.get_time()/5, glfw.get_time()/10, 0.0)
		# Use line 367 instead of 366 if cube should simply rotate arbitrarily in position
		transformLoc = glGetUniformLocation(shader, "transform")
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transformation)
 
		# Draw cube based on triangle polygons from loaded VBO's
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
 
 		# Set rendered texture to window
		glfw.swap_buffers(window)

	# When application loop ends, terminate program
	glfw.terminate()
 

if __name__ == "__main__":
	main()