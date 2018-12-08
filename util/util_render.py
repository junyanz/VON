import os
import numpy as np
import math
import bpy
from mathutils import Vector


def setup_cam_and_lights_old(p0, p1):
    scn = bpy.context.scene

    # set camera view and lightling
    bpy.ops.object.camera_add()
    scn.camera = bpy.data.objects['Camera']

    # Constrain camera to object
    bpy.ops.object.constraint_add(type="TRACK_TO")
    bpy.ops.object.empty_add(type='CUBE', radius=0.1, location=(0, 0, 0))
    scn.camera.constraints["Track To"].target = bpy.data.objects['Empty']
    scn.camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    scn.camera.constraints["Track To"].up_axis = 'UP_Y'

    scn.camera.location = Vector(p1) * 1.2
    scn.camera.data.clip_end = max(p1) * 10   # camera clipping range

    # Set camera light
    bpy.ops.object.lamp_add(type='POINT')
    camlight = bpy.context.active_object
    camlight.name = 'camlight'
    camlight.location = scn.camera.location
    camlight.data.use_specular = False
    camlight.data.energy = 1
    camlight.data.distance = 100
    bpy.ops.object.constraint_add(type="CHILD_OF")
    camlight.constraints['Child Of'].target = scn.camera

    # Set 6 background light, one for each face
    sun_dis = (p1 - p0).max() * 1.25
    center = (p0 + p1) / 2
    sun_locs = np.dot(np.ones((6, 1)), center.reshape(1, 3))
    sun_locs[:3, :] += np.eye(3) * sun_dis
    sun_locs[3:, :] -= np.eye(3) * sun_dis
    print(p0, p1)
    print(sun_locs)
    for i in range(6):
        bpy.ops.object.lamp_add(type='POINT')
        bpy.context.active_object.name = 'facelight_' + str(i)
    sun_counter = 0
    for light in [obj for obj in bpy.data.objects if obj.name[:10] == 'facelight_']:
        light.location = Vector(sun_locs[sun_counter])
        light.data.use_specular = False
        light.data.energy = 0.00
        sun_counter += 1


def render(model_path, render_prefix, az, ele, view_id, res=512):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # load obj
    bpy.ops.import_scene.obj(filepath=model_path, axis_up='Z', axis_forward='-X')
    obj = bpy.context.selected_objects[0]

    # find bounding box
    coords = [(obj.matrix_world * v.co) for v in obj.data.vertices]
    p0 = np.array([min([c.x for c in coords]), min(
        [c.y for c in coords]), min([c.z for c in coords])])
    p1 = np.array([max([c.x for c in coords]), max(
        [c.y for c in coords]), max([c.z for c in coords])])

    setup_cam_and_lights_old(p0, p1)

    scn = bpy.context.scene
    scn.render.resolution_x = 512
    scn.render.resolution_y = 512
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    scn.render.alpha_mode = 'TRANSPARENT'
    scn.render.resolution_percentage = 100
    camera = scn.camera
    # Render

    angle = az * np.pi / 180
    up_angle = ele * np.pi / 180
    x = 2.0 * math.cos(up_angle) * math.cos(angle)
    y = 2.0 * math.cos(up_angle) * math.sin(angle)
    z = 2.0 * math.sin(up_angle)
    camera.location = (x, y, z)

    scn.render.filepath = os.path.join(render_prefix + 'view%03d.jpg' % view_id)
    bpy.ops.render.render(write_still=True)
    print('Image saved: ' + scn.render.filepath)


if __name__ == '__main__':
    import sys
    obj_name = sys.argv[4]
    prefix = (sys.argv[5])
    ele = float(sys.argv[6])
    az = float(sys.argv[7])
    view_id = int(sys.argv[8])
    render(obj_name, prefix, az, ele, view_id)
