import Tkinter as tk
import numpy as np
import tkFont
import os
from PIL import Image

home_dir = os.path.expanduser('~')

images_path = os.path.join(home_dir, 'cloud_images')
data_path = os.path.join(home_dir, 'cloud_data')

image_filenames = [os.path.join(images_path, fname) for fname in os.listdir(images_path)]
width = 500
height = 500
if os.path.isfile(os.path.join(data_path, 'classes.npy')):
    full_class_array = np.load(os.path.join(data_path, 'classes.npy'))
else:
    full_class_array = np.zeros([width, height, len(image_filenames)], dtype=np.int8) - 1

def save_state():
    np.save(os.path.join(data_path, 'classes.npy'), full_class_array)
    np.save(os.path.join(data_path, 'i_image.npy'), np.array(i_image))


def make_stencil(brush_size):
    brush_size = int(brush_size)
    x = np.arange(-brush_size, brush_size+1)[:, None]
    y = np.arange(-brush_size, brush_size+1)[None, :]
    return (x**2 + y**2)**0.5 < brush_size


b1_pressed = False
xold = 0
yold = 0


brush_size = 10.
stencil = make_stencil(brush_size)
current_class = 1
cursor_circle = None
drawing_area = None

if os.path.isfile(os.path.join(data_path, 'i_image.npy')):
    i_image = int(np.load(os.path.join(data_path, 'i_image.npy')))
else:
    i_image = 0


color_list = [
    np.array([0, 0, 255], dtype=np.int16),
    np.array([255, 255, 255], dtype=np.int16),
    np.array([255, 0, 0], dtype=np.int16),
    np.array([0, 255, 0], dtype=np.int16),
]

name_list = [
    'clear',
    'cloudy',
    'doggo',
]

def initialize_overlay():
    image_array_overlay[:, :, :] = image_array_base[:, :, :]
    for i_class, color in enumerate(color_list):
        image_array_overlay[(class_array == i_class)] = color

class_array = full_class_array[:, :, i_image]
image_array_base = np.array(Image.open(image_filenames[i_image]))
image_array_overlay = np.zeros([width, height, 3], dtype=np.uint8)
last_class_array = np.zeros([width, height], dtype=np.int8)
last_overlay_array = np.zeros([width, height, 3], dtype=np.uint8)
initialize_overlay()
last_overlay_array[:] = image_array_overlay[:]
overlay_alpha = 0.5
image = None

root = None

def on_closing():
    save_state()
    root.destroy()

def main():
    global image, drawing_area, root
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', on_closing)
    root.bind('<Control-z>', undo)
    button_font = tkFont.Font(family='Helvetica', size=16, weight='bold')
    drawing_area = tk.Canvas(root, width=width, height=height)
    image = tk.PhotoImage(width=width, height=height)
    drawing_area.create_image((width // 2, height // 2), image=image, state='normal')
    initialize_image()
    drawing_area.grid(row=0, column=0, rowspan=len(name_list))
    drawing_area.bind('<Motion>', motion)
    drawing_area.bind('<ButtonPress-1>', b1down)
    drawing_area.bind('<ButtonRelease-1>', b1up)
    foo = None
    button_size = max([len(name) for name in name_list] + [4])
    for i, name in enumerate(name_list):
        exec('def foo(): return set_class({})'.format(i))
        button = tk.Button(
            root,
            text=name,
            width=button_size,
            font=button_font,
            bg='#{:02x}{:02x}{:02x}'.format(*list(color_list[i])),
            command=foo)
        button.grid(row=i, column=1)
    button = tk.Button(root, text='prev', width=button_size, font=button_font, command=previous_image)
    button.grid(row=len(name_list), column=1)
    button = tk.Button(root, text='next', width=button_size, font=button_font, command=next_image)
    button.grid(row=len(name_list) + 1, column=1)
    stencil_scaler = tk.Scale(root, from_=5, to=50, orient=tk.HORIZONTAL, command=set_stencil_size, length=300)
    stencil_scaler.set(25)
    stencil_scaler.grid(row=len(name_list), column=0)
    root.mainloop()


def undo(event):
    class_array[:] = last_class_array[:]
    image_array_overlay[:] = last_overlay_array[:]
    initialize_image()


def initialize_image():
    image.put(get_tk_string(
        (1 - overlay_alpha) * image_array_base +
        overlay_alpha * image_array_overlay))


def next_image():
    global i_image
    if i_image < len(image_filenames):
        i_image += 1
        reset_canvas()


def previous_image():
    global i_image
    if i_image > 0:
        i_image -= 1
        reset_canvas()


def reset_canvas():
    global image_array_base
    global class_array
    class_array = full_class_array[:, :, i_image]
    image_array_base = np.array(Image.open(image_filenames[i_image]))
    initialize_overlay()
    initialize_image()


def set_stencil_size(size):
    global stencil, brush_size
    brush_size = int(size)
    make_cursor_circle(xold, yold)
    stencil = make_stencil(size)


def set_class(class_number):
    global current_class
    current_class = class_number


def apply_class(x, y):
    x, y = int(x), int(y)
    delta = int(stencil.shape[0] / 2)
    dx_left = max(delta - x, 0)
    dx_right = max(delta - (width - x), 0)
    dy_bottom = max(delta - y, 0)
    dy_top = max(delta - (height - y), 0)
    current_stencil = stencil[
        dx_left:stencil.shape[0]-dx_right-1, dy_bottom:stencil.shape[1]-dy_top-1]
    x_start = x - delta + dx_left
    x_end = x + delta - dx_right
    y_start = y - delta + dy_bottom
    y_end = y + delta - dy_top
    class_array[x_start:x_end, y_start:y_end][current_stencil] = current_class
    image_array_overlay[x_start:x_end, y_start:y_end][current_stencil] = color_list[current_class][None, :]
    image_patch = (1 - overlay_alpha) * image_array_base[x_start:x_end, y_start:y_end, :] + overlay_alpha * image_array_overlay[x_start:x_end, y_start:y_end]
    image.put(get_tk_string(image_patch.astype(np.int16)), to=(x_start, y_start, x_end, y_end))


def get_tk_string(image_array):
    image_array = image_array.astype(np.int16)
    color_string = ''
    for j in range(image_array.shape[1]):
        color_string_list = []
        for i in range(image_array.shape[0]):
            color_string_list.append(
                '#{:02x}{:02x}{:02x}'.format(*list(image_array[i, j, :])))
        color_string += '{' + ' '.join(color_string_list) + '} '
    return color_string[:-1]


def motion(event):
    global xold, yold
    xold = event.x
    yold = event.y
    make_cursor_circle(event.x, event.y)
    if b1_pressed and event.x > 0 and event.y > 0 and event.x < width and event.y < height:
        apply_class(event.x, event.y)


def make_cursor_circle(x, y):
    global cursor_circle
    if cursor_circle is not None:
        drawing_area.delete(cursor_circle)
    cursor_circle = drawing_area.create_oval(
        x + brush_size,
        y + brush_size,
        x - brush_size,
        y - brush_size, outline='white')


def b1down(event):
    global b1_pressed, xold, yold
    b1_pressed = True
    last_class_array[:] = class_array[:]
    last_overlay_array[:] = image_array_overlay[:]
    xold = event.x
    yold = event.y
    apply_class(event.x, event.y)


def b1up(event):
    global b1_pressed
    b1_pressed = False


if __name__ == '__main__':
    main()
