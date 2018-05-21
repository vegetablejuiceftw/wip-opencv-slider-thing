from math import acos, atan, cos, degrees, log, pi, radians, sin, sqrt


def great_circle_distance(lat_1, lng_1, lat_2, lng_2):
    """ Given latitudes and longitudes of two places(in degrees), it will calculate
    their distance between them(on Earth), in meters
    """
    lat_1 = radians(lat_1)
    lat_2 = radians(lat_2)
    lng_1 = radians(lng_1)
    lng_2 = radians(lng_2)
    EARTH_RADIUS = 6356.8 * 1000
    # min(1.0, ...) is needed to avoid rounding errors in case both coordinates are the same. In that case the
    #  argument given to acos() might be very slightly above 1.0, causing acos() to fail
    return EARTH_RADIUS * acos(min(1.0, sin(lat_1) * sin(lat_2) + cos(lat_1) * cos(lat_2) * cos(lng_2 - lng_1)))


def lest_geo(x, y):
    """ Convert L-EST97 coordinates to WGS 84 (used by GPS)

    From http://www.maaamet.ee/rr/geo-lest/
    """
    # Swap the coordinates if necessary
    if x < y:
        x, y = y, x

    a = 6378137.00000000000
    F = 1 / 298.257222100883
    ESQ = (F + F - F * F)
    B0 = ((57.00000000000 + 31.0000000000 / 60.000000000000 + 3.19414800000 / 3600.00000000000) / degrees(1))
    L0 = (24.00000000000 / degrees(1))
    FN = 6375000.00000000000
    FE = 500000.00000000000
    B2 = ((59.00000000000 + 20.00000000000 / 60.00000000000) / degrees(1))
    B1 = (58.00000000000 / degrees(1))
    xx = (x - FN)
    yy = (y - FE)
    t0 = sqrt((1.00000000000 - sin(B0)) / (1.00000000000 + sin(B0)) *
              pow(((1.00000000000 + sqrt(ESQ) * sin(B0)) / (1.00000000000 - sqrt(ESQ) * sin(B0))), sqrt(ESQ)))
    t1 = sqrt((1.00000000000 - sin(B1)) / (1.00000000000 + sin(B1)) *
              pow(((1.00000000000 + sqrt(ESQ) * sin(B1)) / (1.00000000000 - sqrt(ESQ) * sin(B1))), sqrt(ESQ)))
    t2 = sqrt((1.00000000000 - sin(B2)) / (1.00000000000 + sin(B2)) *
              pow(((1.00000000000 + sqrt(ESQ) * sin(B2)) / (1.00000000000 - sqrt(ESQ) * sin(B2))), sqrt(ESQ)))
    m1 = (cos(B1) / pow((1.00000000000 - ESQ * sin(B1) * sin(B1)), 0.50000000000))
    m2 = (cos(B2) / pow((1.00000000000 - ESQ * sin(B2) * sin(B2)), 0.50000000000))
    n1 = ((log(m1) - log(m2)) / (log(t1) - log(t2)))
    FF = (m1 / (n1 * pow(t1, n1)))
    p0 = (a * FF * pow(t0, n1))
    p = pow((yy * yy + (p0 - xx) * (p0 - xx)), 0.50000000000)
    t = pow((p / (a * FF)), (1.00000000000 / n1))
    FII = atan(yy / (p0 - xx))
    LON = (FII / n1 + L0)
    u = ((pi / 2.00000000000) - (2.00000000000 * atan(t)))
    LAT = (u + (ESQ / 2.00000000000 + (5.00000000000 * pow(ESQ, 2) / 24.00000000000) + (pow(ESQ, 3) / 12.00000000000) +
                (13.00000000000 * pow(ESQ, 4) / 360.00000000000)) * sin(2.00000000000 * u) +
           ((7.00000000000 * pow(ESQ, 2) / 48.00000000000) + (29.00000000000 * pow(ESQ, 3) / 240.00000000000) +
            (811.00000000000 * pow(ESQ, 4) / 11520.00000000000)) * sin(4.00000000000 * u) +
           ((7.00000000000 * pow(ESQ, 3) / 120.00000000000) + (81.00000000000 * pow(ESQ, 4) / 1120.00000000000)) *
           sin(6.00000000000 * u) + (4279.00000000000 * pow(ESQ, 4) / 161280.00000000000) * sin(8.00000000000 * u))
    LAT = degrees(LAT)
    LON = degrees(LON)

    return LAT, LON
