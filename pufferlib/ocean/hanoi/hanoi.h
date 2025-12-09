/* Hanoi: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"


// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;


typedef struct {
    int from;
    int to;
} Move;


// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int disks;
    int pegs;
    int tick;
    int max_timesteps;
    Move move;
} Hanoi;

#define OBS(d,p) env->observations[d*env->pegs+p]

void add_log(Hanoi* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Hanoi* env) {
    env->tick = 0;
    memset(env->observations, 0, sizeof(unsigned char)*env->disks*env->pegs);
    for (int i = 0; i < env->disks; i++) {
        OBS(i,0) = 1;
    }
}


static inline void act_to_move(Hanoi *env, int a) {
    int p = a / (env->pegs -1);
    int k = a % (env->pegs -1);
    int q = k;
    if (q >= p) 
        q++;
    env->move.from = p;
    env->move.to = q;
}

static inline bool valid_move(Hanoi *env) {
    int p = env->move.from;
    int q = env->move.to;

    int smallest_p = -1;
    int smallest_q = -1;

    // Find smallest disk on peg p
    for (int d = 0; d < env->disks; d++) {
        if (OBS(d,p) == 1) {
            smallest_p = d;
            break;
        }
    }

    // If p is empty → invalid
    if (smallest_p == -1)
        return false;

    // Find smallest disk on peg q
    for (int d = 0; d < env->disks; d++) {
        if (OBS(d,q) == 1) {
            smallest_q = d;
            break;
        }
    }

    // If q is empty → valid
    if (smallest_q == -1)
        return true;

    // Otherwise: valid only if top disk on p is smaller than top disk on q
    return smallest_p < smallest_q;
}

static inline void move_disk(Hanoi *env) {
    int p = env->move.from;
    int q = env->move.to;

    // find smallest disk on peg p
    int disk = -1;
    for (int d = 0; d < env->disks; d++) {
        if (OBS(d, p) == 1) {
            disk = d;
            break;
        }
    }

    // Must exist because valid_move already checked it is non-empty
    // Remove disk from peg p
    OBS(disk, p) = 0;

    // Place disk on peg q
    OBS(disk, q) = 1;
  }

static inline bool is_goal(Hanoi *env) {
    int goal_peg = env->pegs - 1;

    for (int d = 0; d < env->disks; d++) {
        if (OBS(d, goal_peg) == 0)
            return false;
    }

    return true;
}


// Required function
void c_step(Hanoi* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    act_to_move(env, action); //WARNING: does not check if valid, WARNING: this is only place this is overwritten

    if (valid_move(env)) {
        move_disk(env);
    }

    if (env->tick >= env->max_timesteps) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    if (is_goal(env)) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }
}

// Required function. Should handle creating the client on first call
void c_render(Hanoi* env) {
    if (!IsWindowReady()) {
        InitWindow(800, 600, "PufferLib Hanoi");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int W = GetScreenWidth();
    int H = GetScreenHeight();

    int P = env->pegs;
    int D = env->disks;

    // Peg spacing (evenly spaced across width)
    float peg_spacing = W / (float)(P + 1);

    // Draw pegs
    for (int p = 0; p < P; p++) {
        int x = (int)((p + 1) * peg_spacing);
        DrawLine(x, H * 0.15, x, H * 0.9, RAYWHITE);
    }

    // Disk height and max width
    float disk_h = (H * 0.7) / D;                // stack fits vertically
    float max_disk_w = peg_spacing * 0.8;        // largest disk width

    // Draw disks (from largest to smallest so smaller drawn on top)
    for (int d = D - 1; d >= 0; d--) {
        for (int p = 0; p < P; p++) {
            if (OBS(d,p) == 1) {
                int x_center = (int)((p + 1) * peg_spacing);

                float w = max_disk_w * ((d + 1) / (float)D);   // proportional width
                float h = disk_h * 0.9;

                int stack_index = 0;

                // Count how many disks are below this one on peg p
                for (int dd = d + 1; dd < D; dd++) {
                    if (OBS(dd,p) == 1)
                        stack_index++;
                }

                int y = (int)(H * 0.9 - (stack_index + 1) * disk_h);

                DrawRectangle(
                    x_center - (int)(w / 2),
                    y,
                    (int)w,
                    (int)h,
                    (Color){80 + d*10, 120 + d*10, 200 - d*10, 255}
                );
            }
        }
    }


    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Hanoi* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
