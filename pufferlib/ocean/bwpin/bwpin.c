/* Pure C demo file for Bwpin. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "bwpin.h"

static inline int move_to_action(int from, int to, int pegs) {
      int cells = 2 * pegs + 1;
      if (from < 0 || from >= cells || to < 0 || to >= cells)
          return -1;

      int idx = 0;
      for (int pos = 0; pos < cells; ++pos) {
          if (pos > 0) {
              if (from == pos && to == pos - 1)
                  return idx;
              idx++;
          }
          if (pos < cells - 1) {
              if (from == pos && to == pos + 1)
                  return idx;
              idx++;
          }
      }
      return -1;  // invalid from/to combo
  }


int main() {
    Bwpin env = {0};
    env.pegs = 3;
    int cells = env.pegs * 2 + 1;
    env.observations = calloc((size_t)cells, sizeof(unsigned char));
    env.actions = calloc(1, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));
    env.max_timesteps = 100;
    env.celebrate_ticks = 1;
    env.celebrate_tick = 0;

    int pending_from = -1;
    int pending_to   = -1;
    int action_count = 2 * cells - 2;

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        bool manual_mode = IsKeyDown(KEY_LEFT_SHIFT);

        if (manual_mode) {
            int key;
            while ((key = GetKeyPressed()) != 0) {
                if (key == KEY_BACKSPACE || key == KEY_DELETE) {
                    pending_from = pending_to = -1;
                    continue;
                }

                int cell = -1;
                if (key >= KEY_ONE && key <= KEY_NINE) {
                    cell = key - KEY_ONE;
                } else if (key == KEY_ZERO) {
                    cell = 9;
                } else if (key >= KEY_KP_1 && key <= KEY_KP_9) {
                    cell = key - KEY_KP_1;
                } else if (key == KEY_KP_0) {
                    cell = 9;
                }

                if (cell >= 0 && cell < cells) {
                    if (pending_from < 0) {
                        pending_from = cell;
                        pending_to = -1;
                    } else if (cell != pending_from) {
                        pending_to = cell;
                    }
                }
            }

            env.move.from = pending_from;
            env.move.to   = pending_to;

            if (pending_from < 0 || pending_to < 0) {
                c_render(&env);
                continue;
            }

            int action = move_to_action(pending_from, pending_to, env.pegs);
            if (action < 0) {
                pending_from = pending_to = -1;
                continue;
            }

            env.actions[0] = action;
            pending_from = pending_to = -1;
        } else {
            pending_from = pending_to = -1;
            env.actions[0] = rand() % action_count;
        }

        c_step(&env);
        env.move.from = env.move.to = -1;
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
