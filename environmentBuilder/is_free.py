def is_free(x, y, walls, safetyMargin=1): 

    from CONSTANTS import MAP_WIDTH
    from CONSTANTS import MAP_HEIGHT
    from CONSTANTS import DRONE_RADIUS

    """
    Return True if (x, y) is outside all obstacles and inside the map.
    This function will be replaced to represent the actual environment, i.e. this will be the interface to pybullet
    """

    # 1) Check map bounds (optional, but usually useful)
    if not (0 <= x <= MAP_WIDTH and 0 <= y <= MAP_HEIGHT):
        return False  # treat outside map as invalid
    
    #Get All Wall Locations
    
    for wall in walls:
        if (wall.xmin - safetyMargin*DRONE_RADIUS <= x <= wall.xmax + safetyMargin*DRONE_RADIUS and
            wall.ymin - safetyMargin*DRONE_RADIUS <= y <= wall.ymax + safetyMargin*DRONE_RADIUS
            ):
            return False


    # If none of the obstacles contain the point, it's free
    return True


def is_free_xyz(x, y, z, walls): 

    from CONSTANTS import MAP_WIDTH
    from CONSTANTS import MAP_HEIGHT
    from CONSTANTS import MAP_LENGTH
    from CONSTANTS import DRONE_RADIUS

    """
    Return True if (x, y) is outside all obstacles and inside the map.
    This function will be replaced to represent the actual environment, i.e. this will be the interface to pybullet
    """

    # 1) Check map bounds (optional, but usually useful)
    if not (0 <= x <= MAP_WIDTH and 0 <= y <= MAP_LENGTH and 0<= z <= MAP_HEIGHT):
        return False  # treat outside map as invalid
    
    #Get All Wall Locations
    
    for wall in walls:
        if (wall.xmin - DRONE_RADIUS <= x <= wall.xmax + DRONE_RADIUS and
            wall.ymin - DRONE_RADIUS <= y <= wall.ymax + DRONE_RADIUS and
            wall.zmin - DRONE_RADIUS <= z <= wall.zmax + DRONE_RADIUS
            ):
            return False


    # If none of the obstacles contain the point, it's free
    return True