import taichi as ti
import os

ti.init(ti.gpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1
PI = 3.141592653

# number of planets
N = 1000
# unit mass
m = 1
BH_m = 20000

# galaxy size
galaxy_size = 0.5

# planet radius (for rendering)
planet_radius = 2
BH_radius = 10

Disapper_distance = 0.015
Disapper_distance_BH = 0.01

# init vel
init_vel = 0


# time-step size
h = 1e-5
# substepping
substepping = 3


# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)
Disapper_bool = ti.Vector.field(1, ti.f32, N)
BH_disapper_bool = ti.Vector.field(1, ti.f32, 1)

# pos, vel and force of black hole
BH_pos = ti.Vector.field(2, ti.f32, 2)


BH_vel = ti.Vector.field(2, ti.f32, 2)
#BH_vel[0] = ti.Vector([0,0])

BH_force = ti.Vector.field(2, ti.f32, 2)


@ti.kernel
def initialize():
    center = ti.Vector([0.5, 0.5])

    BH_pos[0] = ti.Vector([0.25, 0.25])
    BH_pos[1] = ti.Vector([0.75, 0.75])

    for i in range(2):
        r = BH_pos[i] - center
        diff = ti.Vector([-r[1], r[0]])
        offset = diff / diff.norm(1e-5)
        BH_vel[i] = offset * 4000

    for i in range(N):
        theta = ti.random() * 2 * PI
        r = (ti.sqrt(ti.random()) * 0.7) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center+offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel
        Disapper_bool = ti.Vector([0])

    BH_disapper_bool[0] = ti.Vector([0])



@ti.kernel
def compute_force():

    # clear force
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    BH_force[0] = ti.Vector([0.0, 0.0])
    BH_force[1] = ti.Vector([0.0, 0.0])

    # compute gravitational force
    for i in range(N):
        if Disapper_bool[i][0] == 0:
            p = pos[i]
            for j in range(i): # bad memory footprint and load balance, but better CPU performance
                if Disapper_bool[i][0] == 0:
                    diff = p-pos[j]
                    r = diff.norm(1e-5)

                    # gravitational force -(GMm / r^2) * (diff/r) for i
                    f = -G * m * m * (1.0/r)**3 * diff

                    # assign to each particle
                    force[i] += f
                    force[j] += -f

            for k in range(2):
                diff_BH = p - BH_pos[k]
                r = diff_BH.norm(1e-5)

                if r < Disapper_distance:
                    Disapper_bool[i] = ti.Vector([1]);
                    vel[i] = ti.Vector([0,0])
                    pos[i] = ti.Vector([-100000,-100000] ) * (ti.random() - 0.5)
                else:
                    f = -G * m * BH_m * (1.0/r)**3 * diff_BH
                    force[i] += f
                    BH_force[k] += -f

                for l in range(k):
                    if BH_disapper_bool[0][0] == 0:
                        diff_inter_BH = BH_pos[k] - BH_pos[l]
                        r_BH = diff_inter_BH.norm(1e-2)

                        if r_BH <= Disapper_distance_BH:
                            BH_disapper_bool[0][0] = 1
                            BH_vel[k] = ti.Vector([0,0])
                            BH_pos[k] = ti.Vector([0.5,0.5])
                            BH_vel[l] = ti.Vector([0,0])
                            BH_pos[l] = ti.Vector([0.5,0.5])
                            BH_force[k] = ti.Vector([0,0])
                            BH_force[l] = ti.Vector([0,0])
                        else:
                            f = -G * BH_m * BH_m * (1.0/r_BH)**3 * diff_inter_BH
                            BH_force[k] += f
                            BH_force[l] += -f

@ti.kernel
def update():
    dt = h/substepping
    for i in range(N):
        if Disapper_bool[i][0] == 0:
            #symplectic euler
            vel[i] += dt*force[i]/m
            pos[i] += dt*vel[i]

    for i in range(2):
        if BH_disapper_bool[0][0] == 0:
            BH_vel[i] += dt*BH_force[i]/BH_m
            BH_vel[i] *= 0.995 # decay
            BH_pos[i] += dt*BH_vel[i]

gui = ti.GUI('N-body problem', (512, 512))
pixels = ti.field(ti.u8, shape=(512, 512, 3))

result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

initialize()

iterations = 1000
for i in range(iterations):

    for j in range(substepping):
        compute_force()
        update()

    gui.clear(0x112F41)

    gui.circles(pos.to_numpy(), color=0xffffff, radius=planet_radius)
    gui.circles(BH_pos.to_numpy(), color=0xff0000, radius=BH_radius)
    # gui.show()

    filename = f'./results/frames/frame_{i:05d}.png'   # create filename with suffix png
    print(f'Frame {i} is recorded in {filename}')
    gui.show(filename)  # export and show in GUI
