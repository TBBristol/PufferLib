/* Bwpin: a sample single-agent grid env.
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
    int pegs;
    int tick;
    int max_timesteps;
    Move move;
    int celebrate_ticks;
    int celebrate_tick;
} Bwpin;

void add_log(Bwpin* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
// white pegs on left (1), black pegs on right (-1) one blank space in middle
void c_reset(Bwpin* env) {
    env->tick = 0;
    memset(env->observations, 0, sizeof(unsigned char)*env->pegs*2 +1);
    for (int i = 0; i < env->pegs; i++) {
        observations[i] = 1;
    }
    for (int i = env->pegs; i < env->pegs*2 +1; i++) {
        observations[i] = -1;
}


static inline void act_to_move(Bwpin *env, int a) {
}

static inline bool valid_move(Bwpin *env) {
}

static inline void move_peg(Bwpin *env) {
}

static inline bool is_goal(Bwpin *env) {
}

/*Following for generating trajectories*/
void goal_set(Bwpin *env) {
//all black pegs on left of white pegs space anywhere
}

void sample_valid_move(Bwpin *env) {
}

void shuffle_moves(Bwpin *env, int n) {
    for (int i = 0; i < n; i++) {
        sample_valid_move(env);
        move_peg(env);
    }
}


// Required function
void c_step(Bwpin* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    act_to_move(env, action); //WARNING: does not check if valid, WARNING: this is only place this is overwritten

    if (valid_move(env)) {
        move_peg(env);
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
        env->celebrate_tick = env->celebrate_ticks; //tells c_render to draw goal
        c_reset(env);
        return;
    }

    env->rewards[0] -= 0.1;
}

// Required function. Should handle creating the client on first call
void c_render(Bwpin* env) {
    if (!IsWindowReady()) {
        InitWindow(800, 600, "PufferLib Bwpin");
        SetTargetFPS(1);      
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int W = GetScreenWidth();
    int H = GetScreenHeight();

    if (env->celebrate_tick > 0) {
          const char *goal_txt = "GOAL!";
          int font = 48;
          int text_w = MeasureText(goal_txt, font);
          DrawText(goal_txt,
                   W / 2 - text_w / 2,
                   (int)(H * 0.45f),
                   font,
                   GOLD);
          env->celebrate_tick--;
      }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Bwpin* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
