#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
import numpy as np


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def polar_line_to_cartesian_seg(line):
    '''Given a line in rho, theta format, return a cartesian seg'''
    assert line.shape == (1, 2)
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return np.array([x1, y1]), np.array([x2, y2])


def line_intersec(line1, line2):
    '''Given a line in rho, theta format, return the intersection
    as x,y coordinates
    '''
    p1, p2 = polar_line_to_cartesian_seg(line1)
    p3, p4 = polar_line_to_cartesian_seg(line2)
    return seg_intersect(p1, p2, p3, p4)


if __name__ == '__main__':
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])

    p3 = np.array([4.0, -5.0])
    p4 = np.array([4.0, 2.0])

    print seg_intersect(p1, p2, p3, p4)

    p1 = np.array([2.0, 2.0])
    p2 = np.array([4.0, 3.0])

    p3 = np.array([6.0, 0.0])
    p4 = np.array([6.0, 3.0])

    print seg_intersect(p1, p2, p3, p4)
