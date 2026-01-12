/* Bwpin: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>


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
    memset(env->observations, 0, sizeof(unsigned char)*(env->pegs*2 +1));
    for (int i = 0; i < env->pegs; i++) {
        env->observations[i] = 1;
    }
    env->observations[env->pegs] = 0;
    for (int i = env->pegs+1; i < 2 * env->pegs +1; i++) {
        env->observations[i] = -1;
    }
    env-> terminals[0] = 0;
    env->rewards[0] = 0;
    env->move.from = env->move.to = -1;
}
static inline void set_move(Bwpin *env, int from, int dir) {
    int cells = env->pegs * 2 + 1;
    int neighbor = from + dir;
    int jump = from + 2 * dir;

    if (neighbor >= 0 && neighbor < cells && env->observations[neighbor] == 0) {
        env->move.from = from;
        env->move.to = neighbor;              // slide into adjacent blank
    } else if (jump >= 0 && jump < cells &&
               env->observations[neighbor] != 0 &&
               env->observations[jump] == 0) {
        env->move.from = from;
        env->move.to = jump;                  // jump over one peg
    } else {
        env->move.from = env->move.to = -1;   // action currently impossible
    }
}

/*- action 0: move peg at cell 0 to the right
  - action 1: move peg at cell 1 to the left
  - action 2: move peg at cell 1 to the right
  - action 3: peg at cell 2 left
  - action 4: peg at cell 2 right
  - …
  - final action: peg at last cell left (since it has no rightward option)
*/

static inline void act_to_move(Bwpin *env, int a) {
      int cells = env->pegs * 2 + 1;
      env->move.from = -1;
      env->move.to = -1;

      int idx = 0;
      for (int pos = 0; pos < cells; ++pos) {
          if (pos > 0 && idx++ == a) {
              set_move(env, pos, -1);
              return;
          }
          if (pos < cells - 1 && idx++ == a) {
              set_move(env, pos, +1);
              return;
          }
      }
  }


static inline bool valid_move(Bwpin *env) {
    int cells = env->pegs * 2 + 1;
    int from = env->move.from;
    int to = env->move.to;

    if (from < 0 || from >= cells || to < 0 || to >= cells || from == to)
        return false;

    unsigned char *board = env->observations;
    if (board[from] == 0 || board[to] != 0)
        return false;

    int delta = to - from;
    if (delta == 1 || delta == -1)
        return true;                   // slide into adjacent blank

    if (delta == 2 || delta == -2) {
        int mid = from + delta / 2;
        return board[mid] != 0;        // jumping over one peg
    }

    return false;                      // longer jumps not allowed
}

static inline void move_peg(Bwpin *env) {
     int from = env->move.from;
     int to = env->move.to;
     if (from < 0 || to < 0)
         return;

     unsigned char peg = env->observations[from];
     env->observations[from] = 0;
     env->observations[to] = peg;
     env->move.from = env->move.to = -1;
}

static inline bool is_goal(Bwpin *env) {
    //all black pegs on left of white pegs space anywhere
    bool seen_white = false;
    int cells = env->pegs * 2 + 1;

    for (int i = 0; i < cells; ++i) {
        unsigned char v = env->observations[i];
        if (v == 1) {
            seen_white = true;
        } else if (v == (unsigned char)-1 && seen_white) {
            return false;              // black found to the right of a white
        }
    }
    return true;

}

/*Following for generating trajectories*/

void goal_set(Bwpin *env) {
      int cells = env->pegs * 2 + 1;
      int blank = rand() % cells;

      // start with all whites
      for (int i = 0; i < cells; ++i) {
          env->observations[i] = 1;
      }
      env->observations[blank] = 0;

      // overwrite the first `pegs` non-blank slots with blacks
      int placed = 0;
      for (int i = 0; i < cells && placed < env->pegs; ++i) {
          if (i == blank) {
              continue;
          }
          env->observations[i] = (unsigned char)-1;
          placed++;
      }

      env->move.from = env->move.to = -1;
  }



void sample_valid_move(Bwpin *env) {
     int cells = env->pegs * 2 + 1;
     static const int deltas[4] = {-1, 1, -2, 2};

     while (1) {
         int from = rand() % cells;
         if (env->observations[from] == 0)
             continue;                        // can't move empty slot

         int start = rand() % 4;
         for (int i = 0; i < 4; ++i) {
             int delta = deltas[(start + i) % 4];
             int to = from + delta;
             if (to < 0 || to >= cells)
                 continue;

             env->move.from = from;
             env->move.to = to;
             if (valid_move(env))
                 return;
         }
     }
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

    int cells = env->pegs * 2 + 1;
    float board_w = W * 0.8f;
    float board_h = H * 0.15f;
    float board_x = (W - board_w) * 0.5f;
    float board_y = H * 0.6f;

    Color wood = (Color){139, 94, 60, 255};
    Color hole = (Color){90, 55, 35, 255};
    Color peg_white = RAYWHITE;
    Color peg_black = BLACK;

    DrawRectangle((int)board_x, (int)board_y, (int)board_w, (int)board_h, wood);

    float spacing = board_w / (cells + 1);
    float radius = board_h * 0.3f;

    for (int i = 0; i < cells; ++i) {
        float cx = board_x + spacing * (i + 1);
        float cy = board_y + board_h * 0.5f;

        DrawCircle((int)cx, (int)cy, radius, hole);

        unsigned char v = env->observations[i];
        if (v == 1) {
            DrawCircle((int)cx, (int)cy, radius * 0.8f, peg_white);
        } else if (v == (unsigned char)-1) {
            DrawCircle((int)cx, (int)cy, radius * 0.8f, peg_black);
        }
    }
    bool manual_mode = IsKeyDown(KEY_LEFT_SHIFT);
    if (manual_mode) {
        const char *msg = (env->move.from < 0)
            ? "Select peg to move"
            : (env->move.to < 0 ? "Select space to move to"
                                : "Release selection or execute move");
        DrawText(msg, 20, H - 80, 24, RAYWHITE);
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
