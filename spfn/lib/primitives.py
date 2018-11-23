import numpy as np
import math
import random

def normalized(v):
    return v / np.linalg.norm(v)

def make_rand_unit_vector(dims=3):
    vec = np.array([random.gauss(0, 1) for i in range(dims)])
    return normalized(vec)

class Plane: # A finite plane patch spanned by x_axis and y_axis
    @staticmethod
    def get_primitive_name():
        return 'plane'

    def __init__(self, n, c, center=None, x_axis=None, y_axis=None, x_range=[-1, 1],  y_range=[-1, 1]):
        if type(n) is not np.ndarray:
            print('Normal {} needs to be a numpy array!'.format(n))
            raise
        # Plane is defined by {p: n^T p = c}, where the bound is determined by xy_range w.r.t. center
        if center is None:
            center = n * c
        self.n = n / np.linalg.norm(n)
        self.c = c
        self.center = center
        self.x_range = x_range
        self.y_range = y_range

        # parameterize the plane by picking axes
        if x_axis is None or y_axis is None:
            ax_tmp = make_rand_unit_vector()
            self.x_axis = normalized(np.cross(ax_tmp, self.n))
            self.y_axis = normalized(np.cross(self.n, self.x_axis))
        else:
            self.x_axis = x_axis
            self.y_axis = y_axis

    def get_area(self):
        return (self.x_range[1]-self.x_range[0])*(self.y_range[1]-self.y_range[0])*np.linalg.norm(np.cross(self.x_axis, self.y_axis))

    def distance_to(self, p): # p should be point as a numpy array
        return abs(np.dot(self.n, p) - self.c)

    def sample_single_point(self, noise_radius=0.0):
        origin = self.center
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        p = origin + x * self.x_axis + y * self.y_axis
        if noise_radius > 0:
            p += random.uniform(0, noise_radius) * make_rand_unit_vector()
        return (p, self.n)
    
    @classmethod
    def create_random(cls, intercept_range=[-1, 1]):
        return cls(make_rand_unit_vector(), random.uniform(*intercept_range))

class Sphere:
    @staticmethod
    def get_primitive_name():
        return 'sphere'

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_area(self):
       return 4 * np.pi * self.radius * self.radius

    def sample_single_point(self):
        n = make_rand_unit_vector()
        p = self.center + self.radius * n
        return (p, n)

class Cylinder:
    @staticmethod
    def get_primitive_name():
        return 'cylinder'

    def __init__(self, center, radius, axis, height=10.0):
        self.center = center
        self.radius = radius
        self.axis = axis
        self.height = height

        tmp_axis = make_rand_unit_vector()
        self.x_axis = normalized(np.cross(tmp_axis, self.axis))
        self.y_axis = normalized(np.cross(self.axis, self.x_axis))

    def get_area(self):
        return 2 * np.pi * self.radius * self.height
    
    def sample_single_point(self):
        kx, ky = make_rand_unit_vector(dims=2)
        n = kx * self.x_axis + ky * self.y_axis
        p = random.uniform(-self.height/2, self.height/2) * self.axis + self.radius * n + self.center
        return (p, n)

class Cone:
    @staticmethod
    def get_primitive_name():
        return 'cone'

    def __init__(self, apex, axis, half_angle, z_min=0.0, z_max=10.0):
        self.apex = apex
        self.axis = axis
        self.half_angle = half_angle
        self.z_min = z_min
        self.z_max = z_max
    
class Box:
    def __init__(self, center, axes, halflengths):
        # axes is 3x3, representing an orthogonal frame
        # sidelength is length-3 array
        self.center = center
        self.axes = axes
        self.halflengths = halflengths

    def get_six_planes(self):
        result = []
        for i, axis in enumerate(self.axes):
            for sgn in range(-1, 2, 2):
                n = sgn * axis
                center = self.center + self.halflengths[i] * n
                c = np.dot(n, center)
                j = (i + 1) % 3
                k = (j + 1) % 3
                x_range = [-self.halflengths[j], self.halflengths[j]]
                y_range = [-self.halflengths[k], self.halflengths[k]]
                plane = Plane(n, c, center=center, x_axis=self.axes[j], y_axis=self.axes[k], x_range=x_range, y_range=y_range)
                result.append(plane)

        return result

    @classmethod
    def create_random(cls, center_range=[-1, 1], halflength_range=[0.5,2]):
        center = np.array([random.uniform(*center_range) for _ in range(3)])
        x_axis = make_rand_unit_vector()
        ax_tmp = make_rand_unit_vector()
        y_axis = normalized(np.cross(ax_tmp, x_axis))
        z_axis = normalized(np.cross(x_axis, y_axis))
        axes = [x_axis, y_axis, z_axis]
        halflengths = [random.uniform(*halflength_range) for _ in range(3)]
        return Box(center, axes, halflengths)

