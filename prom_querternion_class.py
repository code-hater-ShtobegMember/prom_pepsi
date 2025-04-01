import numpy as np


def rotate_vector(v, alpha, beta, gamma):
    """Rotates a 3D vector by these angles for each axis"""
    q = Quaternion
    q.from_euler(alpha, beta, gamma)
    return q.rotate_vector_from_quaternion(q,v)



class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __mul__(self, other):
        """Quaternion multiplication"""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Multiplication only supports Quaternion or scalar values")

    def conjugate(self):
        """Returns the conjugate of the quaternion"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        """Returns the norm of the quaternion"""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        """Returns a normalized quaternion"""
        norm = self.norm()
        if norm == 0:
            raise ZeroDivisionError("Cannot normalize a quaternion with zero norm")
        return self * (1.0 / norm)

    def rotate_vector_from_quaternion(self, v):
        """Rotates a 3D vector using this quaternion"""
        v_quat = Quaternion(0, *v)
        rotated_v = self * v_quat * self.conjugate()
        return np.array([rotated_v.x, rotated_v.y, rotated_v.z])

    @staticmethod
    def from_euler(alpha, beta, gamma):
        """ Create a quaternion from Euler angles (in radians) """
        qx = Quaternion(np.cos(alpha/2), np.sin(alpha/2), 0, 0)
        qy = Quaternion(np.cos(beta/2), 0, np.sin(beta/2), 0)
        qz = Quaternion(np.cos(gamma/2), 0, 0, np.sin(gamma/2))
        return (qz * qy * qx).normalize()

a = [1,0,0]
print(rotate_vector(a,0,0,np.pi/2))
