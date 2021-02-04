from robopy import *
import robopy.base.model as model


# x = UnitQuaternion.Rx(10, 'deg')
# y = UnitQuaternion.Ry(120, 'deg')
# x.animate(y, duration=15)

class link2(SerialLink):
    def __init__(self):
        links = [Revolute(d=0, a=0, alpha=pi / 2, j=0, theta=0, offset=0, qlim=(-160 * pi / 180, 160 * pi / 180)),
                 Revolute(d=0, a=0.4318, alpha=0, j=0, theta=0, offset=0, qlim=(-45 * pi / 180, 225 * pi / 180))]
        colors = graphics.vtk_named_colors(["Red", "DarkGreen", "Blue", "Cyan", "Magenta", "Yellow", "White"])
        file_names = SerialLink._setup_file_names(7)
        param = {
            "cube_axes_x_bounds": np.matrix([[-1.5, 1.5]]),
            "cube_axes_y_bounds": np.matrix([[-0.7, 1.5]]),
            "cube_axes_z_bounds": np.matrix([[-1.5, 1.5]]),
            "floor_position": np.matrix([[0, -0.7, 0]])
        }
        super().__init__(links=links, colors=colors, name='puma_560', stl_files=file_names, param=param)

q1 = np.linspace(1, -180, 500).reshape((-1, 1))
q2 = np.linspace(1, 180, 500).reshape((-1, 1))
q = np.concatenate((q1, q2), axis=1)
robot = link2()
robot.animate(stances=q, frame_rate=30, unit='deg')
# print(robot.fkine(q[0]))