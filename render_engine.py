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


values = [0,0,0,0,0]
offsets = [-1.5, -16, -6, 0, 0]
server_address = 'ws://172.31.68.73:8765'

async def getPositionAndRotation():
	infoNeeded = ['x','y','z','xr','yr']
	for i in range(len(infoNeeded)):
		name = infoNeeded[i]
		async with websockets.connect(server_address) as websocket:
			await websocket.send("get"+name)
			value = await websocket.recv()
			#print("Got " + name + " (" + value + ")")
			values[i] = float(value)

def inputEvent(window, key, scancode, action, mods):
	increment = 0.5
	z_increment = 2
	if action == glfw.PRESS:
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
		elif key == glfw.KEY_ESCAPE:
			glfw.set_window_should_close(window, GL_TRUE)

def createTransformationMatrix(xPos=0, yPos=0, zPos=0, xRotation=0, yRotation=0, zRotation=0):
	rot_x = pyrr.Matrix44.from_x_rotation(xRotation)
	rot_y = pyrr.Matrix44.from_y_rotation(yRotation)
	global_x = pyrr.Matrix44.from_x_rotation(5/180*math.pi)
	#xv = pyrr.Vector3([1.0,0.0,0.0])
	#xv = rot_y * xv
	#rot_x = pyrr.Matrix44.create_from_axis_rotation(xv, xRotation)
	# pyrr.matrix44.create_from_axis_rotation
	transformationMatrix = rot_x @ rot_y
	transformationMatrix[3][0] = xPos
	transformationMatrix[3][1] = yPos
	transformationMatrix[3][2] = zPos

	return transformationMatrix

def main():
	if not glfw.init():
		return
 
	display = [1920, 1080]

	monitors = glfw.get_monitors()
	active_monitor = monitors[1] # 1 = secondary
	window = glfw.create_window(display[0], display[1], "Pyopengl Perspective Projection", active_monitor, None)
 
	if not window:
		glfw.terminate()
		return
 
	glfw.make_context_current(window)
	half_cube_width = 7/2
	#        positions         colors          texture coords
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

	tex_width = 384
	tex_height = 512
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
 
 
	cube = np.array(multi_tex_cube, dtype=np.float32)
 
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
	images_to_load = []
	for i in range(26):
		images_to_load.append("rick_anim/frame_"+str(i)+".jpg")
	#images_to_load = ["rick_astley.jpg", "grass.jpg"]

	images = []
	imgs_data = []

	for im in images_to_load:
		image = Image.open(im)
		images.append(image)
		imgs_data.append(np.array(list(image.getdata()), np.uint8))

	img_id = 0
	image = images[img_id]
	img_data = imgs_data[img_id]
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
	glEnable(GL_TEXTURE_2D)
 
	glUseProgram(shader)
 
	glClearColor(0.0, 0.0, 0.0, 1.0)
	glEnable(GL_DEPTH_TEST)
 
 
	#Creating Projection Matrix
	view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0,6.0,-134.0]))
	view = view @ pyrr.Matrix44.from_x_rotation((6.56/180*math.pi))
	projection = pyrr.matrix44.create_perspective_projection(20, display[0]/display[1], 0.1, 1000.0)
	model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0,0.0,0.0]))

	view_loc = glGetUniformLocation(shader, "view")
	proj_loc = glGetUniformLocation(shader, "projection")
	model_loc = glGetUniformLocation(shader, "model")

	glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

	glfw.set_key_callback(window, inputEvent)
 
	while not glfw.window_should_close(window):

		img_id = int(glfw.get_time()/(1/5))%len(images_to_load)
		image = images[img_id]
		img_data = imgs_data[img_id]
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
		glEnable(GL_TEXTURE_2D)

		glfw.poll_events()
 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
 
		asyncio.get_event_loop().run_until_complete(getPositionAndRotation())

		print(offsets) # -3.6, -7.4, -59
		transformation = createTransformationMatrix(-values[0]+offsets[0], values[1]+offsets[1], -values[2]+offsets[2], -values[3], -values[4], 0.0)
		#transformation = createTransformationMatrix(-values[0]+offsets[0], values[1]+offsets[1], -values[2]+offsets[2], glfw.get_time()/5, glfw.get_time()/10, 0.0)
		# -values[3]-(3/180*math.pi), -values[4]-(2/180*math.pi)

		transformLoc = glGetUniformLocation(shader, "transform")
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transformation)
 
		# Draw Cube
 
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
 
		glfw.swap_buffers(window)
 
	glfw.terminate()
 
 
if __name__ == "__main__":
	main()