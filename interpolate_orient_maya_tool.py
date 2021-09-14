from maya import cmds
from maya import OpenMayaUI as omui
import maya.api.OpenMaya as om

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from shiboken2 import wrapInstance

from distutils.util import strtobool
from functools import partial
import numpy as npy
import math

""" 
Global constants.
"""

# Windows options

INTERP_OPTIONS = ['None',
                  'Lineal',
                  'Hermite',
                  'Catmull-Rom']
ORIENT_OPTIONS = ['None',
                  'Align with curve improved',
                  'Align with curve',
                  'Interpolate rotation']
AXIS_OPTIONS = ['X',
                'Y',
                'Z']
AXIS = {"X": [1.0, 0.0, 0.0],
        "Y": [0.0, 1.0, 0.0],
        "Z": [0.0, 0.0, 1.0]}

# Fileinfo ids to store options

SELECTED_INTERP_FILEINFO = 'selected_interp'
SELECTED_CONSVEL_FILEINFO = 'selected_consvel'
SELECTED_REPARAM_FILEINFO = 'selected_reparam'
SELECTED_NUMFRAMES_FILEINFO = 'selected_numframes'
SELECTED_ORIENT_FILEINFO = 'selected_orient'
SELECTED_TANGENT_FILEINFO = 'selected_tangent'
SELECTED_UP_FILEINFO = 'selected_up'
SELECTED_TENSION_FILEINFO = 'selected_tension'

# Objects attributes.

TRANSLATE_ATTRIBUTES = ['translateX',
                        'translateY',
                        'translateZ']
ROTATE_ATTRIBUTES = ['rotateX',
                     'rotateY',
                     'rotateZ']
VELOCITY_ATTRIBUTES = ['velocityX',
                       'velocityY',
                       'velocityZ']


def inter_lineal(keyframes, frm):
    """
    Function to interpolate linearly
    given keyframes and a frame (t).
    """

    # Save info of time and position of keyframes.
    vt = []
    vx = []
    for i in range(len(keyframes[0])):
        vt.append(keyframes[0][i] / 24.0)
        vx.append(keyframes[1][i])

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
    for i in range(len(keyframes[0])):
        vt.append(keyframes[0][i] / 24.0)
        vx.append(keyframes[1][i])

    # Save keyframes velocities.

    vv = []
    for obj in vel:
        vv.append(obj)

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
    for i in range(len(keyframes[0])):
        vt.append(keyframes[0][i] / 24.0)
        vx.append(keyframes[1][i])

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


def integrate_length(get_pos_x, get_pos_y, get_pos_z, final_frame):
    """
    Function that creates a list with
    length displaced from each frame.
    """

    arclen = [0.0]

    for i in range(final_frame):
        c0 = npy.array([get_pos_x(i), get_pos_y(i), get_pos_z(i)])
        c1 = npy.array([get_pos_x(i + 1), get_pos_y(i + 1), get_pos_z(i + 1)])

        displacement = npy.linalg.norm(c1 - c0)

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

    t_old = fcurv(time=(frame, frame))[0]

    return get_pos_ppa(t_old)


def inter_quaternion(keyframes, idx, frm):
    """
    Function to interpolate rotations as quaternions.
    """

    # Obtain rotation as Euler rotation and convert it
    # to a quaternion.
    # Then save the quaternion rotation and the time of the keyframes.

    vt = []
    vq = []
    for i in range(len(keyframes[0][0])):
        aux = [keyframes[0][1][i],
               keyframes[1][1][i],
               keyframes[2][1][i]]
        q = om.MEulerRotation(math.radians(aux[0]),
                              math.radians(aux[1]),
                              math.radians(aux[2]),
                              om.MEulerRotation.kXYZ).asQuaternion()
        q.normalizeIt()

        vq.append(q)
        vt.append(keyframes[0][0][i] / 24.0)

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
        q = om.MQuaternion.slerp(vq[i - 1], vq[i], u)

    # Return asked coord converted as Euler rotation.

    return math.degrees(q.asEulerRotation()[idx])


def get_tan(func_interp, frm):
    """
    Function to obtain tangent vector
    of function at a time frm.
    """

    f0 = func_interp(frm)
    f1 = func_interp(frm + 1)

    tan = f1 - f0

    return tan


def get_quat_tan(fun_pos_x, fun_pos_y, fun_pos_z, idx, axis, frm):
    """
    Function to obtain a quaternion that
    aligns an axis with the tangent of a function.
    """

    v_axis = om.MVector(AXIS[axis])

    tan = om.MVector([get_tan(fun_pos_x, frm), get_tan(fun_pos_y, frm), get_tan(fun_pos_z, frm)])

    # Return asked coord converted as Euler rotation.

    return math.degrees(om.MQuaternion(v_axis, tan).asEulerRotation()[idx])


def get_control_dir(fun_pos_x, fun_pos_y, fun_pos_z, idx, axis, up_axis, frm):
    """
    Function to obtain a quaternion that
    aligns an axis with the tangent of a function and with its up axis.
    """

    v_axis = om.MVector(AXIS[axis])
    v_up_axis = om.MVector(AXIS[up_axis])

    # First quaternion to orient to tangent.

    euler_rot = om.MEulerRotation(0.0, 0.0, 0.0, om.MEulerRotation.kXYZ)
    for i in [0, 1, 2]:
        euler_rot[i] = math.radians(get_quat_tan(fun_pos_x, fun_pos_y, fun_pos_z, i, axis, frm))

    q1 = euler_rot.asQuaternion()
    q1.normalizeIt()

    v_up_axis = v_up_axis.rotateBy(q1)
    v_up_axis.normalize()

    tan = om.MVector([get_tan(fun_pos_x, frm), get_tan(fun_pos_y, frm), get_tan(fun_pos_z, frm)])

    z = om.MVector([0.0, 0.0, 1.0])

    # Lateral aux vector of the curve.

    l = z ^ tan
    l.normalize()

    # Up aux vector of the curve.

    u = tan ^ l
    u.normalize()

    # And then, compute quaternion to orient up vector.

    q2 = om.MQuaternion(v_up_axis, u)
    q2.normalizeIt()

    # WARNING: This rotation is not completed yet.

    q = (q1 * q2 * q1.conjugate()) * q1

    # Return asked coord converted as Euler rotation.

    return math.degrees(q.asEulerRotation()[idx])


def get_maya_window():
    """
    Function to get the main Maya window as
    a QtGui.QMainWindow instance.
    """

    ptr = omui.MQtUtil.mainWindow()
    if ptr is not None:
        return wrapInstance(int(ptr), QWidget)


class InterpolationWindow(QMainWindow):
    """
    Class to represent an interpolation and orientation
    tool in Maya.
    """

    toolName = 'Interpolation and orientation tool'

    def __init__(self, parent=get_maya_window()):
        self.delete_duplicated_windows()

        super(InterpolationWindow, self).__init__(parent)
        self.setObjectName(self.__class__.toolName)
        self.setWindowTitle('Interpolation and Orientation Tool')

        self.setup_ui()
        self.update_ui()
        self.resize(500, 300)
        self.show()
        self.set_connections_and_events()


    # ---- UI methods -------------------------------------------------

    def setup_ui(self):
        """
        Method to set the ui of the tool
        and store the selected options that control
        interpolation and orientation.
        """

        main_layout = QVBoxLayout()

        """
        INTERPOLATION PANEL
        This panel displays:
         - The selected object which can be interpolated.
         - The different interpolation methods.
         - The Catmull-Rom tension parameter for its interpolation.
         - The constant velocity checkbox to make the interpolation
           velocity constant.
         - The reparametrization checkbox which allows you to control
           velocity of the interpolation once velocity constant is checked.
         - The number of frames on the curve to know when the interpolation ends
           in order to control the velocity of the interpolation.        
        """

        interp_layout = QGridLayout()

        label = QLabel('Interpolate position')
        label.setContentsMargins(0, 0, 0, 10)
        label.setStyleSheet("""font-weight:bold; font-size: 10pt;""")
        interp_layout.addWidget(label, 0, 0)

        label = QLabel('Active objects:     ')
        label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        interp_layout.addWidget(label, 1, 0)

        self.selected_object_label = QLabel(' ')
        self.selected_object_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        interp_layout.addWidget(self.selected_object_label, 1, 1)

        label = QLabel('Type of interpolation:     ')
        label.setAlignment(Qt.AlignLeft)
        interp_layout.addWidget(label, 2, 0)

        self.interp_combo = QComboBox(self)
        self.interp_combo.addItems(INTERP_OPTIONS)
        self.interp_combo.setStyleSheet("""background-color: #363636;""")
        interp_layout.addWidget(self.interp_combo, 2, 1)

        box_layout = QHBoxLayout()
        self.tension = 0
        self.tension_label = QLabel('Catmull-Rom tension: %s' % (self.tension))
        self.tension_label.setAlignment(Qt.AlignLeft)
        box_layout.addWidget(self.tension_label)

        self.tension_slider = QSlider(Qt.Horizontal, self)
        self.tension_slider.setRange(0, 200)
        self.tension_slider.setStyleSheet("""background-color: #363636;""")
        box_layout.addWidget(self.tension_slider)

        widget = QWidget(self)
        widget.setLayout(box_layout)
        interp_layout.addWidget(widget, 3, 1)

        self.checkbox_cons_vel = QCheckBox('Constant Velocity', self)
        interp_layout.addWidget(self.checkbox_cons_vel, 4, 0)

        self.checkbox_reparam = QCheckBox('Reparametrization', self)
        interp_layout.addWidget(self.checkbox_reparam, 4, 1)

        self.end_frame = 0
        self.end_frame_label = QLabel('Number of frames on the curve: %s' % (self.end_frame))
        label.setAlignment(Qt.AlignLeft)
        interp_layout.addWidget(self.end_frame_label, 5, 0)

        # Add interpolation panel to main layout.

        interp_widget = QWidget(self)
        interp_widget.setLayout(interp_layout)
        interp_widget.setStyleSheet("""background-color: #262626; """)
        main_layout.addWidget(interp_widget)

        """
        ORIENTATION PANEL
        This panel displays:
         - The different options to orient the object.
         - The tangent axis options used when align curve is selected.
         - The up axis options used when align with curve improved is selected.
        """

        orient_layout = QGridLayout()
        label = QLabel('Orientation')
        label.setContentsMargins(0, 0, 0, 10)
        label.setStyleSheet("""font-weight:bold; font-size: 10pt;""")
        orient_layout.addWidget(label, 0, 0)

        label = QLabel('Type of orientation: ')
        orient_layout.addWidget(label, 1, 0)

        self.orient_combo = QComboBox(self)
        self.orient_combo.addItems(ORIENT_OPTIONS)
        self.orient_combo.setStyleSheet("""background-color: #363636;""")
        orient_layout.addWidget(self.orient_combo, 1, 1)

        label = QLabel('Tangent axis: ')
        orient_layout.addWidget(label, 2, 0)

        self.tangent_axis_combo = QComboBox(self)
        self.tangent_axis_combo.addItems(AXIS_OPTIONS)
        self.tangent_axis_combo.setStyleSheet("""background-color: #363636;""")
        orient_layout.addWidget(self.tangent_axis_combo, 2, 1)

        label = QLabel('Up axis: ')
        orient_layout.addWidget(label, 3, 0)

        self.up_axis_combo = QComboBox(self)
        self.up_axis_combo.addItems(AXIS_OPTIONS)
        self.up_axis_combo.setStyleSheet("""background-color: #363636;""")
        orient_layout.addWidget(self.up_axis_combo, 3, 1)

        # Add orientation panel to main layout.

        orient_widget = QWidget(self)
        orient_widget.setLayout(orient_layout)
        orient_widget.setStyleSheet("""background-color: #262626; """)
        main_layout.addWidget(orient_widget)

        # Button which will launch execution of interpolations and orientations.

        self.interpolate_orient_btn = QPushButton('Interpolate and orient', self)
        main_layout.addWidget(self.interpolate_orient_btn)

        # Finally, set main layout to the window.

        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def update_ui(self):
        """
        Method to read selected options saved in the scene
        as fileInfo and set their values to the ui.
        """

        sel_info = cmds.fileInfo(SELECTED_INTERP_FILEINFO, query=True)
        if len(sel_info) != 0: self.interp_combo.setCurrentText(sel_info[0])
        sel_info = cmds.fileInfo(SELECTED_CONSVEL_FILEINFO, query=True)
        if len(sel_info) != 0: self.checkbox_cons_vel.setChecked(strtobool(sel_info[0]))
        sel_info = cmds.fileInfo(SELECTED_REPARAM_FILEINFO, query=True)
        if len(sel_info) != 0: self.checkbox_reparam.setChecked(strtobool(sel_info[0]))
        sel_info = cmds.fileInfo(SELECTED_NUMFRAMES_FILEINFO, query=True)
        if len(sel_info) != 0:
            self.end_frame = int(float(sel_info[0]))
            self.end_frame_label.setText('Number of frames on the curve: %s' % (self.end_frame))
            self.end_frame_label.repaint()
        sel_info = cmds.fileInfo(SELECTED_ORIENT_FILEINFO, query=True)
        if len(sel_info) != 0: self.orient_combo.setCurrentText(sel_info[0])
        sel_info = cmds.fileInfo(SELECTED_TANGENT_FILEINFO, query=True)
        if len(sel_info) != 0: self.tangent_axis_combo.setCurrentText(sel_info[0])
        sel_info = cmds.fileInfo(SELECTED_UP_FILEINFO, query=True)
        if len(sel_info) != 0: self.up_axis_combo.setCurrentText(sel_info[0])
        sel_info = cmds.fileInfo(SELECTED_TENSION_FILEINFO, query=True)
        if len(sel_info) != 0:
            self.tension = float(sel_info[0])
            self.tension_label.setText('Catmull-Rom tension: %s' % (self.tension))
            self.tension_label.repaint()
            self.tension_slider.setValue(self.tension * 100)

        # If an object is selected, update this on the ui also.

        self.update_selected_obj()

    def update_tension_label(self):
        """
        Method to update Catmull-Rom tension
        information on the ui when its value is changed
        and store its value.
        """

        self.tension = self.tension_slider.value() / 100.0
        self.tension_label.setText('Catmull-Rom tension: %s' % (self.tension))
        self.tension_label.repaint()

    def update_end_frame(self):
        """
        Method to update when the
        animation ends on the ui when
        its value is changed.
        """

        self.end_frame_label.setText('Number of frames on the curve: %s' % (self.end_frame))
        self.end_frame_label.repaint()

    def update_selected_obj(self):
        """
        Method to update the selected object
        information on the ui and create needed keyable
        attributes for different interpolations.
        """

        self.selected_object = cmds.ls(selection=True, transforms=True)[-1] if len(cmds.ls(selection=True, transforms=True)) > 0 else ' '
        self.selected_object_label.setText(self.selected_object)
        self.selected_object_label.repaint()

        # This attribute is used when constant velocity and reparam checkbox are checked
        # and stores as keyframes when we want the animation to end, controlling its velocity.

        if not cmds.objExists('%s.reparametrization' % (self.selected_object)) and self.selected_object is not ' ':
            cmds.addAttr(self.selected_object,
                         longName='reparametrization',
                         niceName='Reparametrization',
                         shortName='reparam',
                         keyable=True,
                         attributeType='double',
                         min=0.0)

        # This attribute is used when Hermite interpolation is selected.
        # Each position keyframe needs a velocity keyframe too.

        for i in [0, 1, 2]:
            if not cmds.objExists('%s.%s' % (self.selected_object, VELOCITY_ATTRIBUTES[i])) and self.selected_object is not ' ':
                cmds.addAttr(self.selected_object,
                             longName=VELOCITY_ATTRIBUTES[i],
                             keyable=True,
                             attributeType='double')

    def set_connections_and_events(self):
        """
        Method to connect functions with events.
        """

        # Connection when the catmull-rom tension slider is changed.

        self.tension_slider.valueChanged.connect(self.update_tension_label)

        # Connection when an object is selected using a scriptJob.

        self.select_scriptJob_ID = cmds.scriptJob(event=['SelectionChanged', self.update_selected_obj])

        # Connection when the button panel is clicked to execute interpolations and interpolations.

        self.interpolate_orient_btn.clicked.connect(self.interpolate_and_orient)

    def delete_duplicated_windows(self):
        """
        Method to delete a duplicated window
        when the script is launched.
        """

        for obj in get_maya_window().children():
            if obj.objectName() == self.__class__.toolName:
                obj.setParent(None)
                obj.deleteLater()

    def deleteLater(self):
        """
        Method to delete the scriptJob created
        and save the current panel configuration
        when the window is deleted.
        """

        if cmds.scriptJob(exists=self.select_scriptJob_ID):
            cmds.scriptJob(kill=self.select_scriptJob_ID, force=True)
        self.save_tool_config()

    def closeEvent(self, event):
        """
        Method to delete the scriptJob created
        and save the current panel configuration
        when the window is closed.
        """

        if cmds.scriptJob(exists=self.select_scriptJob_ID):
            cmds.scriptJob(kill=self.select_scriptJob_ID, force=True)
        self.save_tool_config()
        event.accept()

    def save_tool_config(self):
        """
        Method to save the current panel configuration
        in the scene as a fileInfo.
        """

        cmds.fileInfo(SELECTED_INTERP_FILEINFO, self.interp_combo.currentText())
        cmds.fileInfo(SELECTED_CONSVEL_FILEINFO, self.checkbox_cons_vel.isChecked())
        cmds.fileInfo(SELECTED_REPARAM_FILEINFO, self.checkbox_reparam.isChecked())
        cmds.fileInfo(SELECTED_NUMFRAMES_FILEINFO, self.end_frame)
        cmds.fileInfo(SELECTED_ORIENT_FILEINFO, self.orient_combo.currentText())
        cmds.fileInfo(SELECTED_TANGENT_FILEINFO, self.tangent_axis_combo.currentText())
        cmds.fileInfo(SELECTED_UP_FILEINFO, self.up_axis_combo.currentText())
        cmds.fileInfo(SELECTED_TENSION_FILEINFO, self.tension)


    # ---- Driver methods -------------------------------------------------

    def create_driver_node(self, transform_type):
        """
        Method to create a node as a driver which stores
        original keyframes and interpolated keyframes and
        connects interpolated keyframes with the correspondent
        transform attributes of the object.
        """

        # Different driver attributes considering rotation or translation.

        if transform_type == 'rotate':
            att_type = 'doubleAngle'
            transform_attributes = ROTATE_ATTRIBUTES
        elif transform_type == 'translate':
            att_type = 'double'
            transform_attributes = TRANSLATE_ATTRIBUTES
        else:
            att_type = 'double'
            transform_attributes = TRANSLATE_ATTRIBUTES

        # Delete the driver to reset interpolated keyframes.

        self.remove_driver_node(transform_type)

        # Create the driver with its attributes if it doesn't exist.

        driver_node_name = '%s_%s_driver' % (self.selected_object, transform_type)
        if not cmds.objExists(driver_node_name):
            cmds.createNode('network', name=driver_node_name, skipSelect=True)

            for i in [0, 1, 2]:
                # This attribute stores interpolated keyframes.

                driver_attribute = 'interpolated_%s' % (transform_attributes[i])
                cmds.addAttr(driver_node_name,
                             longName=driver_attribute,
                             keyable=True,
                             attributeType=att_type)

                # This attribute stores original keyframes.

                cmds.addAttr(driver_node_name,
                             longName=transform_attributes[i],
                             keyable=True,
                             attributeType=att_type)

        # Update animcurves connections considering the driver.
        # Now the translation or rotation attributes of the object
        # will be receiving the interpolated keyframes from the driver.

        for i in [0, 1, 2]:
            connections = cmds.listConnections('%s.%s' % (self.selected_object, transform_attributes[i]), type="animCurve")

            if connections is not None:
                cmds.disconnectAttr('%s.output' % (connections[0]), '%s.%s' % (self.selected_object, transform_attributes[i]))
                cmds.connectAttr('%s.output' % (connections[0]), '%s.%s' % (driver_node_name, transform_attributes[i]), force=True)
                cmds.connectAttr('%s.interpolated_%s' % (driver_node_name, transform_attributes[i]), '%s.%s' % (self.selected_object, transform_attributes[i]), lock=True)

        return driver_node_name

    def remove_driver_node(self, transform_type):
        """
        Method to reconnect original keyframes with
        the correspondent transform attributes of the object
        and delete the driver.
        """

        # Different driver attributes considering rotation or translation.

        if transform_type == 'rotate':
            transform_attributes = ROTATE_ATTRIBUTES
        elif transform_type == 'translate':
            transform_attributes = TRANSLATE_ATTRIBUTES
        else:
            transform_attributes = TRANSLATE_ATTRIBUTES

        # Reconnect original keyframes with the correspondent transform attributes.

        driver_node_name = '%s_%s_driver' % (self.selected_object, transform_type)
        if cmds.objExists(driver_node_name):
            for i in [0, 1, 2]:
                connections = cmds.listConnections('%s.%s' % (driver_node_name, transform_attributes[i]), type="animCurve")

                if connections is not None:
                    cmds.setAttr('%s.%s' % (self.selected_object, transform_attributes[i]), lock=False)
                    cmds.disconnectAttr('%s.output' % (connections[0]), '%s.%s' % (driver_node_name, transform_attributes[i]))
                    cmds.disconnectAttr('%s.interpolated_%s' % (driver_node_name, transform_attributes[i]), '%s.%s' % (self.selected_object, transform_attributes[i]))
                    cmds.connectAttr('%s.output' % (connections[0]), '%s.%s' % (self.selected_object, transform_attributes[i]), force=True)

            # And finally, delete the driver.

            cmds.delete(driver_node_name)

    # ---- Interpolation methods -------------------------------------------------

    def get_end_frame(self):
        """
        Method to get the last frame of the animation considering
        the x, y and z channels of translation.
        """

        end_frame = 0

        # Consider getting original keyframes or interpolated keyframes.

        driver_node_name = '%s_translate_driver' % (self.selected_object)
        obj = driver_node_name if cmds.objExists(driver_node_name) else self.selected_object

        for i in [0, 1, 2]:
            channel_end_frame = cmds.keyframe(obj,
                                              attribute=TRANSLATE_ATTRIBUTES[i],
                                              query=True,
                                              timeChange=True)

            if channel_end_frame is not None:
                end_frame = channel_end_frame[-1] if channel_end_frame[-1] > end_frame else end_frame

        return int(end_frame)

    def interpolate_and_orient(self):
        """
        Main method of this class to interpolate and
        orient the selected object.
        """

        if self.selected_object is ' ':
            return

        # Obtain selected panel options.

        interpolation = str(self.interp_combo.currentText())
        orientation = str(self.orient_combo.currentText())
        tension = self.tension
        cons_vel = self.checkbox_cons_vel.isChecked()
        self.end_frame = self.get_end_frame()
        axis = self.tangent_axis_combo.currentText()
        up_axis = self.up_axis_combo.currentText()
        reparametrization = self.checkbox_reparam.isChecked()

        # Obtain reparametrization animCurve as a partial function.

        values_variable_change = partial(cmds.keyframe, self.selected_object, attribute='reparametrization', query=True, eval=True)

        # Initialize position, rotation and velocities keyframe vectors,
        # and interpolation and orientation functions (one for each coordinate).

        keyframes = [[None for i in range(2)] for j in range(3)]
        r_keyframes = [[None for i in range(2)] for j in range(3)]
        velocities = [None] * 3
        func = [None] * 3
        r_func = [None] * 3

        """
        Interpolation of translation keyframes of the object.
        """

        if interpolation == 'None':
            self.remove_driver_node('translate')
        else:
            driver_node_name = self.create_driver_node('translate')
            for i in [0, 1, 2]:

                # Save the different keyframes of the animation.

                keyframes[i][0] = [int(a) for a in cmds.keyframe(driver_node_name,
                                                                 attribute=TRANSLATE_ATTRIBUTES[i],
                                                                 query=True,
                                                                 timeChange=True)]

                keyframes[i][1] = cmds.keyframe(driver_node_name,
                                                attribute=TRANSLATE_ATTRIBUTES[i],
                                                query=True,
                                                valueChange=True)

                velocities[i] = cmds.keyframe(self.selected_object,
                                              attribute=VELOCITY_ATTRIBUTES[i],
                                              query=True,
                                              valueChange=True)

                # Create interpolation functions.

                if interpolation == 'Lineal':
                    func[i] = partial(inter_lineal, keyframes[i])
                elif interpolation == 'Hermite':
                    func[i] = partial(inter_hermite, keyframes[i], velocities[i])
                elif interpolation == 'Catmull-Rom':
                    func[i] = partial(inter_catrom, keyframes[i], tension)

            # If making constant velocity, create arclen table.

            if cons_vel:
                arclen_table = integrate_length(func[0], func[1], func[2], self.end_frame)

                # Total number of frames in ppa curve.

                length_table = arclen_table[- 1]

                self.end_frame = int(length_table / (1.0 / 24.0))

                # Assign functions to set ppa curve.

                for i in [0, 1, 2]:
                    func[i] = partial(get_pos_ppa, func[i], arclen_table)

                # Reparametrize curve controlling velocity.

                if reparametrization:
                    for i in [0, 1, 2]:
                        func[i] = partial(get_pos_reparam, func[i], values_variable_change)

            # Upload translation drivers.

            for i in [0, 1, 2]:
                driver_attribute = '%s.interpolated_%s' % (driver_node_name, TRANSLATE_ATTRIBUTES[i])
                self.interpolate_keyframes(driver_attribute, keyframes[i], func[i])

        """
        Interpolation of rotation keyframes and orientation of the object.
        """

        if orientation == 'None':
            self.remove_driver_node('rotate')
        else:
            driver_node_name = self.create_driver_node('rotate')
            for i in [0, 1, 2]:
                # Save the different keyframes of the animation.

                r_keyframes[i][0] = [int(a) for a in cmds.keyframe(driver_node_name,
                                                                   attribute=ROTATE_ATTRIBUTES[i],
                                                                   query=True,
                                                                   timeChange=True)]

                r_keyframes[i][1] = cmds.keyframe(driver_node_name,
                                                  attribute=ROTATE_ATTRIBUTES[i],
                                                  query=True,
                                                  valueChange=True)
            # Create orientation functions.

            for i in [0, 1, 2]:
                if orientation == 'Interpolate rotation':
                    r_func[i] = partial(inter_quaternion, r_keyframes, i)
                    if cons_vel and interpolation != 'None':
                        r_func[i] = partial(get_pos_ppa, r_func[i], arclen_table)
                        if reparametrization:
                            r_func[i] = partial(get_pos_reparam, r_func[i], values_variable_change)
                elif interpolation != 'None':
                    if orientation == 'Align with curve':
                        r_func[i] = partial(get_quat_tan, func[0], func[1], func[2], i, axis)
                    elif orientation == 'Align with curve improved':
                        r_func[i] = partial(get_control_dir, func[0], func[1], func[2], i, axis, up_axis)

            # Upload orientation drivers.

            if orientation == 'Interpolate rotation' or interpolation != 'None':
                for i in [0, 1, 2]:
                    driver_attribute = '%s.interpolated_%s' % (driver_node_name, ROTATE_ATTRIBUTES[i])
                    self.interpolate_keyframes(driver_attribute, r_keyframes[i], r_func[i])

            # In case there is nothing to interpolate or orient.

            else:
                self.remove_driver_node('rotate')

        # Display new end frame of animation on the ui.

        self.update_end_frame()

    def interpolate_keyframes(self, name, keyframes, func):
        """
        Method to compute interpolated frames from keyframes
        of a specific attribute.
        """

        end_frame = self.end_frame if self.end_frame > keyframes[0][-1] else keyframes[0][-1]
        for frame in range(keyframes[0][0], end_frame + 1):
            cmds.setKeyframe(name, value=func(frame), time=frame)


if __name__ == '__main__':
    window = InterpolationWindow()