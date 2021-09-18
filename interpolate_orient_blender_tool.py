import bpy
from functools import partial
from mathutils import Vector, Quaternion
import numpy as npy

""" 
Global constant.
"""

AXIS = {
    'X': [1.0, 0.0, 0.0],
    'Y': [0.0, 1.0, 0.0],
    'Z': [0.0, 0.0, 1.0]}


def inter_lineal(keyframes, frm):
    """
    Function to interpolate linearly
    given keyframes and a frame (t).
    """

    # Save info of time and position of keyframes.

    vt = []
    vx = []
    for obj in keyframes.keyframe_points:
        vt.append(obj.co[0] / 24.0)
        vx.append(obj.co[1])

    # Compute time in seconds.

    t = frm / 24.0

    # Index to know in which part of the curve we are.

    i = 0
    while (i < len(vt)) and (vt[i] < t):
        i = i + 1

    # If not allowed to interpolate.

    if i == 0:
        x = vx[0]
    elif i == len(vt):
        x = vx[i - 1]

    # Apply interpolation.

    else:
        u = (t - vt[i - 1]) / (vt[i] - vt[i - 1])
        x = vx[i - 1] + u * (vx[i] - vx[i - 1])

    return x


def inter_hermite(keyframes, vel, frm):
    """
    Function to interpolate using Hermite
    given keyframes, velocities and a frame(t).
    """

    # Save info of time and position of keyframes.

    vt = []
    vx = []
    for obj in keyframes.keyframe_points:
        vt.append(obj.co[0] / 24.0)
        vx.append(obj.co[1])

    # Save keyframes velocities.

    vv = []
    for obj in vel.keyframe_points:
        vv.append(obj.co[1])

    # Compute time in seconds.

    t = frm / 24.0

    # Index to know in which part of curve we are.

    i = 0
    while (i < len(vt)) and (vt[i] < t):
        i = i + 1

    # Cases where interpolation is not allowed.

    if i == 0:
        x = vx[0]
    elif i == len(vt):
        x = vx[i - 1]

    # Compute interpolation.

    else:
        u = (t - vt[i - 1]) / (vt[i] - vt[i - 1])
        x = (1 - 3 * u * u + 2 * u * u * u) * vx[i - 1] + u * u * (3 - 2 * u) * vx[i] + u * (u - 1) * (u - 1) * vv[
            i - 1] + u * u * (u - 1) * vv[i]

    return x


def inter_catrom(keyframes, ten, frm):
    """
    Function to interpolate using Catmull-Rom
    given keyframes, tension and a frame (t).
    """

    # Save info of time and position of keyframes.

    vt = []
    vx = []
    for obj in keyframes.keyframe_points:
        vt.append(obj.co[0] / 24.0)
        vx.append(obj.co[1])

    # Compute time in seconds.

    t = frm / 24.0

    # Position in trajectory part we are.

    i = 1
    while (i < len(vt)) and (vt[i] < t):
        i = i + 1

    # If all keyframes passed.

    if i == len(vt):
        MuP = vx[i - 1]

    # Apply interpolation.

    else:
        u = (t - vt[i - 1]) / (vt[i] - vt[i - 1])
        Mu = npy.array(
            [-ten * u + 2 * ten * (u * u) - ten * (u * u * u), 1 + (ten - 3) * (u * u) + (2 - ten) * (u * u * u),
             ten * u + (3 - 2 * ten) * (u * u) + (ten - 2) * (u * u * u), -ten * (u * u) + ten * (u * u * u)])

        # Different cases where pick points to interpolate.

        if (i == 1) and (i == len(vt) - 1):
            pos = npy.array([vx[i - 1], vx[i - 1], vx[i], vx[i]])

        elif i == 1:
            pos = npy.array([vx[i - 1], vx[i - 1], vx[i], vx[i + 1]])

        elif i == len(vt) - 1:
            pos = npy.array([vx[i - 2], vx[i - 1], vx[i], vx[i]])

        else:
            pos = npy.array([vx[i - 2], vx[i - 1], vx[i], vx[i + 1]])

        MuP = npy.dot(Mu, pos)

    return MuP


def slerp(q1, q2, u):
    """
    Function to interpolate spherically
    given two vectors and a param u.
    """

    theta = npy.arccos(npy.dot(q1, q2))

    qu = [None] * 4
    qu = ((npy.sin(theta * (1 - u)) / npy.sin(theta)) * npy.array(q1)) + (
            (npy.sin(theta * u) / npy.sin(theta)) * npy.array(q2))

    return qu


def module(vector):
    """
    Function to obtain the 
    module of a vector.
    """

    squared_sum = 0.0

    for obj in vector:
        squared_sum += obj * obj

    return npy.sqrt(squared_sum)


def norm(vector):
    """
    Function to normalize a vector.
    """

    n = [0] * len(vector)
    mod = module(vector)

    if mod != 0:
        for i in range(len(vector)):
            n[i] = vector[i] / mod

    return n


def inter_quaternion(keyframes, idx, frm):
    """
    Function to interpolate quaternions.
    """

    # Save the four coordinates of each quaternion and the time of the keyframes.

    vt = []
    vq = []
    for i in range(len(keyframes[0].keyframe_points)):
        aux = [keyframes[0].keyframe_points[i].co[1],
               keyframes[1].keyframe_points[i].co[1],
               keyframes[2].keyframe_points[i].co[1],
               keyframes[3].keyframe_points[i].co[1]]
        vq.append(norm(aux))
        vt.append(keyframes[0].keyframe_points[i].co[0] / 24.0)

    # Compute time in seconds.

    t = frm / 24.0

    # Position in trajectory part we are.

    i = 1
    while (i < len(vt)) and (vt[i] < t):
        i = i + 1

    # If all keyframes passed.

    if i == len(vt):
        q = vq[i - 1]

    # If haven't arrived to first frame.

    elif t < vt[0]:
        q = vq[0]

    # Apply interpolation.

    else:
        u = (t - vt[i - 1]) / (vt[i] - vt[i - 1])
        q = slerp(vq[i - 1], vq[i], u)

    # Return asked coord.

    return q[idx]


def get_tan(func_interp, frm):
    """
    Function to obtain tangent vector 
    of function at a time frm.
    """

    f0 = func_interp(frm)
    f1 = func_interp(frm + 1)

    tan = f1 - f0

    return tan


def get_quaternion(e, t, idx):
    """
    Function to obtain a quaternion that
    rotates vector e to vector t.
    """

    e = norm(e)
    t = norm(t)

    v = npy.cross(e, t)
    v = norm(v)

    theta = npy.arccos(npy.dot(e, t))

    sintheta = npy.sin(theta / 2)

    q = [npy.cos(theta / 2), sintheta * v[0], sintheta * v[1], sintheta * v[2]]

    return q[idx]


def get_quat_tan(fun_pos_x, fun_pos_y, fun_pos_z, idx, axis, frm):
    """
    Function to obtain a quaternion that
    aligns an axis with the tangent of a function.
    """

    v_axis = AXIS[axis]

    tan = [get_tan(fun_pos_x, frm), get_tan(fun_pos_y, frm), get_tan(fun_pos_z, frm)]

    return get_quaternion(v_axis, tan, idx)


def get_control_dir(fun_pos_x, fun_pos_y, fun_pos_z, idx, axis, up_axis, frm):
    """
    Function to obtain a quaternion that
    aligns an axis with the tangent of a function and with its up axis.
    """

    v_axis = Vector(AXIS[axis])
    v_up_axis = Vector(AXIS[up_axis])

    # First quaternion to orient to tangent.

    q1 = Quaternion((0.0, 0.0, 0.0, 0.0))
    for i in [0, 1, 2, 3]:
        q1[i] = get_quat_tan(fun_pos_x, fun_pos_y, fun_pos_z, i, axis, frm)
    q1.normalize()

    v_up_axis.rotate(q1)
    v_up_axis.normalize()

    tan = [get_tan(fun_pos_x, frm), get_tan(fun_pos_y, frm), get_tan(fun_pos_z, frm)]

    z = Vector((0.0, 0.0, 1.0))

    # Lateral aux vector of the curve.

    l = Vector(npy.cross(tan, z))
    l.normalize()

    # Up aux vector of the curve.

    u = Vector(npy.cross(l, tan))
    u.normalize()

    # And then, compute quaternion to orient up vector.

    q2 = Quaternion((0.0, 0.0, 0.0, 0.0))
    for i in [0, 1, 2, 3]:
        q2[i] = get_quaternion(v_up_axis, u, i)
    q2.normalize()

    q1.rotate(q2)

    return q1[idx]


def integrate_length(get_pos_x, get_pos_y, get_pos_z, final_frame):
    """
    Function that creates a list with
    length displaced from
    each frame.
    """

    arclen = [0.0]

    for i in range(final_frame):
        c0 = npy.array([get_pos_x(i), get_pos_y(i), get_pos_z(i)])
        c1 = npy.array([get_pos_x(i + 1), get_pos_y(i + 1), get_pos_z(i + 1)])

        displacement = module(c1 - c0)

        # Each position is a frame and its value is the distance until that moment.

        arclen.append(arclen[i] + displacement)

    return arclen


def get_frame(length, table):
    """
    Function to obtain in which frame a length
    is reached.
    """

    i = 1
    while i < len(table) and length >= table[i]:
        i = i + 1

    if i >= len(table):
        i = i - 1

    frame = i - 1 + (length - table[i - 1]) / (table[i] - table[i - 1] + 1e-5)

    return frame


def get_pos_ppa(func_interp, table, frm):
    """
    Function to obtain correspondent
    frame when length is reached.
    """

    desired_length = frm / 24.0

    orig_frm = get_frame(desired_length, table)

    return func_interp(orig_frm)


def get_pos_reparam(get_pos_ppa, fcurv, frame):
    """
    Function that transforms a curve with constant
    velocity to have the desired velocity.
    """

    t_old = fcurv.evaluate(frame)

    return get_pos_ppa(t_old)


class InterpolateOrientOperator(bpy.types.Operator):
    """
    Class to compute interpolations and orientations when
    the panel button is pressed.
    """

    bl_label = 'Interpolate and Orient'
    bl_idname = 'dh.interpolate_orient_operator'
    bl_description = 'Interpolate and orient the object with the action selected'

    # Error function.

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    # When the button is pressed...

    def execute(self, context):

        # Save items selected of action, interpolation and tension.

        action = context.scene.item_actions
        interpolation = context.scene.item_interpolations
        tension = context.scene.tension
        orientation = context.scene.item_orientations
        axis = context.scene.item_axis
        up_axis = context.scene.item_up_axis
        cons_vel = context.scene.constant_velocity
        reparametrization = context.scene.reparametrization
        values_variable_change = bpy.data.actions[action].fcurves.find('value_variable_change')

        # Save number of frames in the action.

        end_frame = context.scene.end_frame = int(bpy.data.actions[action].frame_range[1])

        # Initialize keyframe vectors, velocities, interpolation functions
        # and drivers name (one for each coordinate).

        keyframes = [None] * 3
        velocities = [None] * 3
        func = [None] * 3
        q_keyframes = [None] * 4
        q_func = [None] * 4
        drv_name = [None] * 3

        if interpolation == 'NONE':
            bpy.context.object.driver_remove('location', -1)
        else:
            for i in [0, 1, 2]:

                # Save the number of frames in the action.

                keyframes[i] = bpy.data.actions[action].fcurves.find('location', index=i)
                velocities[i] = bpy.data.actions[action].fcurves.find('velocity', index=i)

                # Create interpolation functions and driver name.

                if interpolation == 'LINEAL':
                    func[i] = partial(inter_lineal, keyframes[i])
                    drv_name[i] = context.object.name + '_lineal_loc_' + str(i)
                elif interpolation == 'HERMITE':
                    func[i] = partial(inter_hermite, keyframes[i], velocities[i])
                    drv_name[i] = context.object.name + '_hermite_loc_' + str(i)
                elif interpolation == 'CATROM':
                    func[i] = partial(inter_catrom, keyframes[i], tension)
                    drv_name[i] = context.object.name + '_catrom_loc_' + str(i)

            # If making constant velocity, create arclen table.

            if cons_vel:
                arclen_table = integrate_length(func[0], func[1], func[2], end_frame)

                # Total number of frames in ppa curve.

                length_table = arclen_table[len(arclen_table) - 1]
                context.scene.end_frame = int(length_table / (1.0 / 24.0))

                # Assign functions to set ppa curve.

                for i in [0, 1, 2]:
                    func[i] = partial(get_pos_ppa, func[i], arclen_table)

                # Reparametrize curve controlling velocity.

                if reparametrization:
                    for i in [0, 1, 2]:
                        func[i] = partial(get_pos_reparam, func[i], values_variable_change)

            for i in [0, 1, 2]:
                # Upload driver and register in its correspondent position.

                bpy.app.driver_namespace[drv_name[i]] = func[i]
                d = bpy.context.object.driver_add('location', i).driver
                d.expression = drv_name[i] + '(frame)'

        if orientation == '3' or (interpolation == 'NONE' and orientation != '0'):
            bpy.context.object.driver_remove('rotation_quaternion', -1)
        else:
            for i in range(4):
                # Save info of frames of the action.

                q_keyframes[i] = bpy.data.actions[action].fcurves.find('rotation_quaternion', index=i)

            for i in range(4):
                if orientation == '0':
                    q_func[i] = partial(inter_quaternion, q_keyframes, i)
                    if cons_vel and interpolation != 'NONE':
                        q_func[i] = partial(get_pos_ppa, q_func[i], arclen_table)
                        if reparametrization:
                            q_func[i] = partial(get_pos_reparam, q_func[i], values_variable_change)
                    drv_name = context.object.name + '_inter_quaternion_' + str(i)
                elif interpolation != 'NONE':
                    if orientation == '1':
                        q_func[i] = partial(get_quat_tan, func[0], func[1], func[2], i, axis)
                        drv_name = context.object.name + '_orient_quaternion_' + str(i)
                    elif orientation == '2':
                        q_func[i] = partial(get_control_dir, func[0], func[1], func[2], i, axis, up_axis)
                        drv_name = context.object.name + '_control_quaternion_' + str(i)

                # Upload driver and register in its correspondent position.

                bpy.app.driver_namespace[drv_name] = q_func[i]
                d = bpy.context.object.driver_add('rotation_quaternion', i).driver
                d.expression = drv_name + '(frame)'

        return {'FINISHED'}


class InterpolateOrientPanel(bpy.types.Panel):
    """
    Class to represent the panel in Blender.
    """

    bl_label = 'Interpolate and orientate'
    bl_idname = 'OBJECT_PT_interp_action'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context):

        layout = self.layout
        obj = context.object

        # ----------------PANEL OF INTERPOLATION OF POSITION----------------#

        box = layout.box()
        row = box.row(align=True)
        row.label(text='Interpolate action', icon='CURVE_BEZCURVE')

        # Name of selected object.

        split = box.split(factor=0.3, align=True)
        split.label(text='Active object: ')
        split.label(text=obj.name)

        # Two lists: one of actions and another one of different interpolations as declared below.

        row = box.column()
        row.prop(data=context.scene, property='item_actions')
        row.prop(data=context.scene, property='item_interpolations')

        # Coordinates of velocity and value of tension.

        row = box.row()
        if context.scene.item_interpolations == 'HERMITE':
            row.prop(obj, 'velocity')
        elif context.scene.item_interpolations == 'CATROM':
            split = row.split(factor=0.3, align=True)
            col = split.column()
            col = split.column()
            col.prop(data=context.scene, property='tension')

        row = box.row()
        split = row.split(factor=0.5, align=True)
        col = split.column()
        col.prop(data=context.scene, property='constant_velocity')
        col.label(text='Number of frames on the curve: ' + str(context.scene.end_frame))

        col = split.column()
        col.prop(data=context.scene, property='reparametrization')

        if context.scene.reparametrization:
            col.prop(data=context.object, property='value_variable_change')

        # ----------------PANEL OF INTERPOLATION OF QUATERNIONS AND ORIENT----------------#

        box = layout.box()
        row = box.row(align=True)
        row.label(text='Orientation of the object:', icon='MESH_MONKEY')

        # Axis to orient and type of interpolation.

        row = box.column()
        row.prop(data=context.scene, property='item_orientations')
        if (context.scene.item_orientations == '1') or (context.scene.item_orientations == '2'):
            row.prop(data=context.scene, property='item_axis')
            if context.scene.item_orientations == '2':
                row.prop(data=context.scene, property='item_up_axis')

        # Button/operator.

        row = layout.row()
        row.operator('dh.interpolate_orient_operator')


def register():
    bpy.utils.register_class(InterpolateOrientOperator)
    bpy.utils.register_class(InterpolateOrientPanel)

    bpy.types.Scene.item_actions = bpy.props.EnumProperty(name='Action', description='Action to interpolate',
                                                          items=[(obj.name, obj.name, '') for obj in bpy.data.actions],
                                                          update=None)

    bpy.types.Scene.item_interpolations = bpy.props.EnumProperty(name='Interpolation',
                                                                 description='Interpolation method to use',
                                                                 items=[('NONE', 'None', ''),
                                                                        ('LINEAL', 'Lineal', ''),
                                                                        ('HERMITE', 'Hermite', ''),
                                                                        ('CATROM', 'Catmull-Rom', '')], default='NONE')

    bpy.types.Scene.tension = bpy.props.FloatProperty(name='Tension',
                                                      description='Parameter to use Catmull-Rom interpolation',
                                                      default=0.5, min=0.0, max=2.0)

    bpy.types.Object.velocity = bpy.props.FloatVectorProperty(name='Velocity', description='Velocity of the object')

    bpy.types.Scene.item_axis = bpy.props.EnumProperty(name='Tangent axis',
                                                       description='Axis that will be used to orient the object',
                                                       items=[('Z', 'Z', 'Z axis'), ('Y', 'Y', 'Y axis'),
                                                              ('X', 'X', 'X axis')], default='X')

    bpy.types.Scene.item_up_axis = bpy.props.EnumProperty(name='Up axis',
                                                          description='Auxiliar axis to orient the object',
                                                          items=[('Z', 'Z', 'Z axis'), ('Y', 'Y', 'Y axis'),
                                                                 ('X', 'X', 'X axis')], default='Z')

    bpy.types.Scene.item_orientations = bpy.props.EnumProperty(name='Type of orientation',
                                                               description='How the object will be oriented',
                                                               items=[
                                                                   ('3', 'None', "Don't interpolate orientation"),
                                                                   ('2', 'Align with curve improved',
                                                                    'Controlling orientation of the object depending on selected up axis'),
                                                                   ('1', 'Align with curve',
                                                                    'Orient the object through the curve'),
                                                                   ('0', 'Interpolate quaternions',
                                                                    'Only interpolate quaternions')],
                                                               default='3')

    bpy.types.Object.value_variable_change = bpy.props.FloatProperty(name='Variable change',
                                                                     description='Value of variable change',
                                                                     default=0.0)

    bpy.types.Scene.reparametrization = bpy.props.BoolProperty(name='Reparametrization',
                                                               description='Reparametrization of the curve',
                                                               default=False)

    bpy.types.Scene.constant_velocity = bpy.props.BoolProperty(name='Constant Velocity',
                                                               description='Set constant velocity to the object',
                                                               default=False)

    bpy.types.Scene.end_frame = bpy.props.IntProperty(name='Number of frames on the curve',
                                                      description='Number of frames till the end of the curve')


def unregister():
    bpy.utils.unregister_class(InterpolateOrientOperator)
    bpy.utils.unregister_class(InterpolateOrientPanel)


if __name__ == '__main__':
    register()