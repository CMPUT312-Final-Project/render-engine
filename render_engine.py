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


values = [0,0,0]

async def getPositionAndRotation():
	infoNeeded = ['x','y','z']
	for i in range(len(infoNeeded)):
		name = infoNeeded[i]
		async with websockets.connect('ws://172.31.73.76:8765') as websocket:
			await websocket.send("get"+name)
			value = await websocket.recv()
			#print("Got " + name + " (" + value + ")")
			values[i] = float(value)

def createTransformationMatrix(xPos=0, yPos=0, zPos=0, xRotation=0, yRotation=0, zRotation=0):
	rot_x = pyrr.Matrix44.from_x_rotation(xRotation)
	rot_y = pyrr.Matrix44.from_y_rotation(yRotation)
	rot_z = pyrr.Matrix44.from_y_rotation(zRotation)
	transformationMatrix = rot_x * rot_y * rot_z
	transformationMatrix[3][0] = xPos
	transformationMatrix[3][1] = yPos
	transformationMatrix[3][2] = zPos

	return transformationMatrix

def main():
	if not glfw.init():
		return
 
	display = [1920, 1080]

	monitors = glfw.get_monitors()
	window = glfw.create_window(display[0], display[1], "Pyopengl Perspective Projection", monitors[1], None)
 
	if not window:
		glfw.terminate()
		return
 
	glfw.make_context_current(window)
	#        positions         colors          texture coords
	cube = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			-0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			-0.5, 0.5, -0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			-0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			-0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			-0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			-0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			-0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
 
			0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
			-0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			-0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0]
	# convert to 32bit float
 
 
	cube = np.array(cube, dtype=np.float32)
 
	indices = [0, 1, 2, 2, 3, 0,
			   4, 5, 6, 6, 7, 4,
			   8, 9, 10, 10, 11, 8,
			   12, 13, 14, 14, 15, 12,
			   16, 17, 18, 18, 19, 16,
			   20, 21, 22, 22, 23, 20]
 
	indices = np.array(indices, dtype=np.uint32)
 
	VERTEX_SHADER = """
 
			  #version 330
 
			  in vec3 position;
			  in vec3 color;
			  in vec2 InTexCoords;
 
			  out vec3 newColor;
			  out vec2 OutTexCoords;
 
			  uniform mat4 transform; 
			  
			  
			  uniform mat4 view;
			  uniform mat4 model;
			  uniform mat4 projection;
 
			  void main() {
 
				gl_Position = projection * view * model * transform * vec4(position, 1.0f);
			   newColor = color;
			   OutTexCoords = InTexCoords;
 
				}
 
 
		  """
 
	FRAGMENT_SHADER = """
		   #version 330
 
			in vec3 newColor;
			in vec2 OutTexCoords;
 
			out vec4 outColor;
			uniform sampler2D samplerTex;
 
		   void main() {
 
			  outColor = texture(samplerTex, OutTexCoords);
 
		   }
 
	   """
 
	# Compile The Program and shaders
 
	shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
											  OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
 
 
	VBO = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, VBO)
	glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)
 
 
	# Create EBO
	EBO = glGenBuffers(1)
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)
 
 
	# get the position from  shader
	position = glGetAttribLocation(shader, 'position')
	glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(0))
	glEnableVertexAttribArray(position)
 
 
	color = glGetAttribLocation(shader, 'color')
	color = 1 # This is very hacky
	glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(12))
	glEnableVertexAttribArray(color)
 
 
	texCoords = glGetAttribLocation(shader, "InTexCoords")
	glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 8, ctypes.c_void_p(24))
	glEnableVertexAttribArray(texCoords)
 
	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
 
 
	# Set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
 
 
	# Set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
 
 
	# load image
	image = Image.open("rick_astley.jpg")
	img_data = np.array(list(image.getdata()), np.uint8)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
	glEnable(GL_TEXTURE_2D)
 
	glUseProgram(shader)
 
	glClearColor(0.0, 0.0, 0.0, 1.0)
	glEnable(GL_DEPTH_TEST)
 
 
	#Creating Projection Matrix
	view =pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0,0.0,-100.0] ))
	projection = pyrr.matrix44.create_perspective_projection(20.0, display[0]/display[1], 0.1, 100.0)
	model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0,0.0,0.0]))

	view_loc = glGetUniformLocation(shader, "view")
	proj_loc = glGetUniformLocation(shader, "projection")
	model_loc = glGetUniformLocation(shader, "model")

	glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
 
 
	while not glfw.window_should_close(window):
		glfw.poll_events()
 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
 
		asyncio.get_event_loop().run_until_complete(getPositionAndRotation())

		print(values)
		transformation = createTransformationMatrix(values[0], values[1], values[2], 0.5*glfw.get_time(),0.8*glfw.get_time(),0.0)

		transformLoc = glGetUniformLocation(shader, "transform")
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transformation)
 
		# Draw Cube
 
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
 
		glfw.swap_buffers(window)
 
	glfw.terminate()
 
 
if __name__ == "__main__":
	main()