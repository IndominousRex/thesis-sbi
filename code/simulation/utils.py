import jax
import jax.numpy as jnp

## WGS-84 constants
equator_radius = 6378137  # semi-major axis
polar_radius = 6356752.314245  # semi-minor axis
eccentricity2 = (
    1 - polar_radius**2 / equator_radius**2
)  # Square of eccentricity


def R_x(angle):
    """
    Elementary rotation around x-axis.
    """

    c_a = jnp.cos(angle)
    s_a = jnp.sin(angle)

    return jnp.array([[1, 0, 0], [0, c_a, s_a], [0, -s_a, c_a]])


def R_y(angle):
    """
    Elementary rotation around y-axis.
    """

    c_a = jnp.cos(angle)
    s_a = jnp.sin(angle)

    return jnp.array([[c_a, 0, -s_a], [0, 1, 0], [s_a, 0, c_a]])


def R_z(angle):
    """
    Elementary rotation around z-axis.
    """

    c_a = jnp.cos(angle)
    s_a = jnp.sin(angle)

    return jnp.array([[c_a, s_a, 0], [-s_a, c_a, 0], [0, 0, 1]])


def ENU_VelTransform(geo_position):
    """
    Transformation matrix for ENU-velocity to time derivative of geodetic coordinates.
    position = [latitude (degrees), longitude (degrees), height (meters)]
    """

    geo_position = jnp.radians(geo_position)

    N = equator_radius / jnp.sqrt(
        1 - eccentricity2 * jnp.sin(geo_position[0]) ** 2
    )
    M = (
        N
        * (1 - eccentricity2)
        / (1 - eccentricity2 * jnp.sin(geo_position[0]) ** 2)
    )

    return jnp.array(
        [
            [180 / jnp.pi / (M + geo_position[2]), 0, 0],
            [
                0,
                180
                / jnp.pi
                / ((N + geo_position[2]) * jnp.cos(geo_position[0])),
                0,
            ],
            [0, 0, 1],
        ]
    )


def rotate_2D_Vector(x, y, angle):
    """
    Rotation of 2D vector.
    """

    rotation_matrix = jnp.array(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )

    return rotation_matrix @ jnp.array([x, y])


def geodetic_to_ecef(lat, lon, h):
    """
    Converts geodetic coordinates (latitude, longitude, height) to
    Earth-Centered, Earth-Fixed (ECEF) coordinates.

    Args:
        lat (float or jnp.ndarray): Latitude in degrees. Can be a scalar or an array.
        lon (float or jnp.ndarray): Longitude in degrees. Can be a scalar or an array.
        h (float or jnp.ndarray): Height above the ellipsoid in meters.
            Can be a scalar or an array.

    Returns:
        jnp.ndarray: ECEF coordinates as an array of shape (..., 3),
        where each entry contains the [x, y, z] coordinates in meters.

    Notes:
        - This function supports broadcasting for inputs of varying shapes,
          as long as they are compatible.
    """

    lat, lon = jnp.radians(lat), jnp.radians(lon)
    N = equator_radius / jnp.sqrt(1 - eccentricity2 * jnp.sin(lat) ** 2)
    x = (N + h) * jnp.cos(lat) * jnp.cos(lon)
    y = (N + h) * jnp.cos(lat) * jnp.sin(lon)
    z = (N * (1 - eccentricity2) + h) * jnp.sin(lat)

    return jnp.stack([x, y, z], axis=-1)


def ecef_to_geodetic(x, y, z):
    """
    Converts ECEF coordinates to geodetic coordinates using Ferrari's solution.

    Args:
        x (float or jnp.ndarray): ECEF x-coordinate (meters).
        y (float or jnp.ndarray): ECEF y-coordinate (meters).
        z (float or jnp.ndarray): ECEF z-coordinate (meters).

    Returns:
        jnp.ndarray: Geodetic coordinates as [latitude (degrees), 
        longitude (degrees), height (meters)].
    """

    # WGS-84 constants
    e_prime2 = (
        equator_radius**2 / polar_radius**2
    ) - 1  # Second eccentricity squared

    # Longitude
    lon = jnp.arctan2(y, x)

    # Compute intermediate variables
    p = jnp.sqrt(x**2 + y**2)
    r = jnp.sqrt(p**2 + z**2)

    # Ferrari's solution for latitude
    E = (equator_radius - polar_radius) / equator_radius
    F = (54 * polar_radius**2 * z**2) / r**4
    G = 1 + (eccentricity2**2 * p**2) / (r**2 * z**2)
    c = (E**2 * G) / (1 + jnp.sqrt(1 + F))
    s = jnp.cbrt(1 + c + jnp.sqrt(c**2 + 2 * c))
    P = F / (3 * s)
    Q = jnp.sqrt(1 + 2 * eccentricity2**2 * p**2 / ((1 + s + P) ** 2 * z**2))
    r0 = equator_radius / jnp.sqrt(1 + Q * eccentricity2**2)
    u = jnp.sqrt((r - r0) ** 2 + z**2)

    # Compute geodetic latitude
    lat = jnp.arctan2(
        z + e_prime2 * polar_radius * jnp.sin(u) ** 3,
        p - eccentricity2 * equator_radius * jnp.cos(u) ** 3,
    )

    # Compute height
    N = equator_radius / jnp.sqrt(1 - eccentricity2 * jnp.sin(lat) ** 2)
    h = p / jnp.cos(lat) - N

    return jnp.stack([jnp.degrees(lat), jnp.degrees(lon), h], axis=-1)


def geodetic_to_enu(geo_positions, geo_reference):
    """
    Convert geodetic coordinates to ENU local coordinates.

    Args:
        positions (jnp.ndarray): Array of shape (..., 3), where each entry is
            [latitude (degrees), longitude (degrees), height (meters)].
        reference (jnp.ndarray): Array of shape (3,), the reference
            [latitude (degrees), longitude (degrees), height (meters)].

    Returns:
        jnp.ndarray: Array of shape (..., 3) containing the local ENU coordinates
        for each input position, in meters.
    """

    # Convert reference position to ECEF
    ref_lat, ref_lon, ref_h = geo_reference
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_h)

    # Compute rotation matrix from ECEF to ENU
    sin_lat, cos_lat = (
        jnp.sin(jnp.radians(ref_lat)),
        jnp.cos(jnp.radians(ref_lat)),
    )
    sin_lon, cos_lon = (
        jnp.sin(jnp.radians(ref_lon)),
        jnp.cos(jnp.radians(ref_lon)),
    )

    R = jnp.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )

    # Convert positions to ECEF
    ecef_positions = geodetic_to_ecef(
        geo_positions[..., 0], geo_positions[..., 1], geo_positions[..., 2]
    )

    # Compute relative ECEF vectors
    delta_ecef = ecef_positions - ref_ecef

    # Rotate into ENU
    enu_coords = jnp.einsum("ij,...j->...i", R, delta_ecef)

    return enu_coords


def enu_to_geodetic(enu_positions, reference):
    """
    Transforms ENU positions into geodetic positions (latitude, longitude, height).

    Args:
        enu_positions (jnp.ndarray): Array of shape (..., 3), where each entry is
            [E, N, U] (positions in ENU coordinates in meters).
        reference (jnp.ndarray): Array of shape (3,), the reference position
            [latitude (radians), longitude (radians), height (meters)].

    Returns:
        jnp.ndarray: Array of shape (..., 3), where each entry contains
        [latitude, longitude, height] in radians (latitude, longitude) and meters (height).
    """

    # Extract reference geodetic coordinates
    lat_ref, lon_ref, h_ref = reference
    ecef_ref = geodetic_to_ecef(lat_ref, lon_ref, h_ref)

    # Rotation matrix from ECEF to ENU
    sin_lat, cos_lat = jnp.sin(lat_ref), jnp.cos(lat_ref)
    sin_lon, cos_lon = jnp.sin(lon_ref), jnp.cos(lon_ref)

    R_enu_to_ecef = jnp.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0, cos_lat, sin_lat],
        ]
    )

    # Convert ENU to ECEF
    ecef_positions = (
        jnp.einsum("ij,...j->...i", R_enu_to_ecef, enu_positions) + ecef_ref
    )

    # Apply conversion
    geo_positions = ecef_to_geodetic(
        ecef_positions[..., 0], ecef_positions[..., 1], ecef_positions[..., 2]
    )

    return geo_positions


def ned_to_body_velocity(v_n, v_e, v_d, roll, pitch, yaw):
    """
    Converts velocity from NED coordinates to body-fixed coordinates.

    Parameters:
    - v_n: float
        North velocity component
    - v_e: float
        East velocity component
    - v_d: float
        Down velocity component
    - roll: float
        Roll angle in radians
    - pitch: float
        Pitch angle in radians
    - yaw: float
        Yaw angle in radians

    Returns:
    - ndarray, shape (3,)
        Velocity vector in body-fixed coordinates [v_forward, v_right, v_down]
    """
    # Convert inputs to JAX array
    V_ned = jnp.array([v_n, v_e, v_d])

    # Compose rotation matrix from NED to body frame
    # Order: yaw -> pitch -> roll (R_x * R_y * R_z)
    R_n_b = R_x(roll) @ R_y(pitch) @ R_z(yaw)

    # Compute the body-fixed velocity vector
    V_body = R_n_b @ V_ned

    return V_body


@jax.jit
def point_in_polygon(points, polygon):
    """
    Determines if a set of 2D points is inside a convex polygon.

    Args:
        points (jnp.ndarray): Array of 2D points of shape (M, 2).
        polygon (jnp.ndarray): A set of 2D points defining a convex polygon of
            shape (N, 2) where the vertices are ordered (clockwise or counterclockwise).

    Returns:
        jnp.ndarray: A boolean array of shape (M,), where True indicates that the
        corresponding point is inside (or on the edge) of the polygon.
    """

    # ensure points is a 2D array
    points = jnp.atleast_2d(points)

    # Compute the "next" vertex for each vertex of the polygon
    next_polygon = jnp.roll(polygon, shift=-1, axis=0)

    # Compute edges of the polygon: shape (N, 2)
    edges = next_polygon - polygon

    # Compute vectors from each polygon vertex to each point:
    # Resulting shape is (M, N, 2) by broadcasting.
    vectors = points[:, None, :] - polygon[None, :, :]

    # Compute the z-component of the cross products for each edge with its corresponding vector:
    # This produces an array of shape (M, N)
    crosses = (
        edges[None, :, 0] * vectors[..., 1]
        - edges[None, :, 1] * vectors[..., 0]
    )

    # A point is inside if all cross products for that point are non-negative or all non-positive.
    inside = jnp.logical_or(
        jnp.all(crosses >= 0, axis=1), jnp.all(crosses <= 0, axis=1)
    )

    return inside

