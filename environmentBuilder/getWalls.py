import numpy as np
from CONSTANTS import DEFAULT_WALL_THICKNESS, MAP_HEIGHT, MAP_WIDTH, PILLAR_L, SEED
from environmentBuilder.is_free import is_free

class Wall:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    rgbaColor: list
    color: str

def createWall(x1,x2,y1,y2,z1=0,z2=1):
    wall = Wall()
    wall.xmin = min(x1,x2)
    wall.xmax = max(x1,x2)
    wall.ymin = min(y1,y2)
    wall.ymax = max(y1,y2)
    wall.zmin = min(z1,z2)
    wall.zmax = max(z1,z2)
    wall.rgbaColor=[0.7, 0.7, 0.7, 1]
    wall.color = 'grey'
    return wall

def createVerticalWall(x,y1,y2):
    wall = createWall(x-DEFAULT_WALL_THICKNESS/2, x+DEFAULT_WALL_THICKNESS/2, y1, y2)
    return wall

def createHorizontalWall(x1,x2,y):
    wall = createWall(x1, x2, y-DEFAULT_WALL_THICKNESS/2, y+DEFAULT_WALL_THICKNESS/2)
    return wall

def createPillar(x,y,z=0):
    wall = Wall()
    wall.xmin = x - PILLAR_L/2
    wall.xmax = x + PILLAR_L/2
    wall.ymin = y - PILLAR_L/2
    wall.ymax = y + PILLAR_L/2
    wall.zmin = z
    wall.zmax = z+1
    wall.rgbaColor = [1, 0, 0, 1]
    wall.color = 'red'
    return wall

def getLayout(layout_id="easy",random_obstacles="none",bounding_walls=True,blockingPositions=[]):

    #np.random.seed(SEED)

    layout = []

    if layout_id == "easy":

        walls = [
            createVerticalWall(4, 0, 8),
            createVerticalWall(10, 16, 8),
            createHorizontalWall(8, 12, 8),
        ]

        for wall in walls:
            layout.append(wall)

    if layout_id == "medium":

        walls = [
            createVerticalWall(6, 0, 6),
            createVerticalWall(8, 10, 16),
            createVerticalWall(12, 3, 6),
            createVerticalWall(3, 3, 6),
            createVerticalWall(4, 9, 14),
            createHorizontalWall(8, 10, 10),
            createHorizontalWall(13, 16, 10),
            createHorizontalWall(3, 12, 6),
        ]

        for wall in walls:
            layout.append(wall)

    if layout_id == "difficult":

        walls = [
            createVerticalWall(6, 4, 8),
            createVerticalWall(12, 13, 8),
            createVerticalWall(10, 2, 5),
            createVerticalWall(13, 0, 3),
            createVerticalWall(3, 0, 5),
            createVerticalWall(4, 11, 16),
            createHorizontalWall(4, 8, 11),
            createHorizontalWall(10, 16, 5),
            createHorizontalWall(0, 12, 8),
            createHorizontalWall(12, 8, 13),
        ]

        for wall in walls:
            layout.append(wall)

    if bounding_walls:

        walls = [
            createVerticalWall(0, 0, 16),
            createVerticalWall(16, 0, 16),
            createHorizontalWall(0, 16, 0),
            createHorizontalWall(0, 16, 16)
        ]

        for wall in walls:
            layout.append(wall)

    if random_obstacles != "none":

        if random_obstacles == "sparse":
            randomObstacleCount = 5
        elif random_obstacles == "crowded":
            randomObstacleCount = 10
    
        walls = []
        for i in range(0,randomObstacleCount):
            x = np.random.random()*MAP_WIDTH
            y = np.random.random()*MAP_HEIGHT

            pillar = createPillar(x,y)

            #print(blockingPositions)

            if not blockingPositions or all(
                is_free(px, py, [pillar], safetyMargin=1.5) for px, py, _ in blockingPositions #chnage to is free_xyz if doing 3D
            ):
                walls.append(pillar)

        for wall in walls:
            layout.append(wall)
 
    return layout

def distance_to_obstacle(p, wall):
    dx = max(wall.xmin - p[0], 0, p[0] - wall.xmax)
    dy = max(wall.ymin - p[1], 0, p[1] - wall.ymax)
    dz = max(wall.zmin - p[2], 0, p[2] - wall.zmax)

    return np.sqrt(dx*dx + dy*dy + dz*dz)


def clearance_along_path(positions, walls):
    clearance = np.zeros(len(positions))
    for i, position in enumerate(positions):
        distances = []
        for wall in walls:
            distances.append(distance_to_obstacle(position, wall))
        clearance[i] = min(distances)

    return clearance
