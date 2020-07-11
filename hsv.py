import math
def HSVDistance(hsv_1,hsv_2):
    H_1,S_1,V_1 = hsv_1
    H_2,S_2,V_2 = hsv_2
    R = 100
    angle = 30
    h = R * math.cos(angle / 180 * math.pi)
    r = R * math.sin(angle / 180 * math.pi)
    x1 = r * V_1 * S_1 * math.cos(H_1 / 180 * math.pi);
    y1 = r * V_1 * S_1 * math.sin(H_1 / 180 * math.pi);
    z1 = h * (1 - V_1);
    x2 = r * V_2 * S_2 * math.cos(H_2 / 180 * math.pi);
    y2 = r * V_2 * S_2 * math.sin(H_2 / 180 * math.pi);
    z2 = h * (1 - V_2);
    dx = x1 - x2;
    dy = y1 - y2;
    dz = z1 - z2;
    return math.sqrt(dx * dx + dy * dy + dz * dz);
