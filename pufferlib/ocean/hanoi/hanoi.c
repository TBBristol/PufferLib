/* Pure C demo file for Hanoi. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "hanoi.h"

static inline int move_to_action(int from, int to, int pegs) {
  int k = (to > from) ? to - 1 : to;
  return from * (pegs - 1) + k;
}

int main() {
    Hanoi env = {0};
    env.disks = 3;
    env.pegs = 3;
    size_t obs_count = (size_t)env.disks * env.pegs;
    env.observations = calloc(obs_count, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.max_timesteps = 100;
    env.tick = 0;

    int pending_from = -1;
    int pending_to   = -1;

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

              int peg = -1;
              if (key >= KEY_ONE && key <= KEY_NINE) {
                  peg = key - KEY_ONE;          // '1' → peg 0
              } else if (key == KEY_ZERO) {
                  peg = 9;                      // allow '0' for peg 10 if needed
              } else if (key >= KEY_KP_1 && key <= KEY_KP_9) {
                  peg = key - KEY_KP_1;
              } else if (key == KEY_KP_0) {
                  peg = 9;
              }

              if (peg >= 0 && peg < env.pegs) {
                  if (pending_from < 0) {
                      pending_from = peg;
                      pending_to = -1;
                  } else if (peg != pending_from) {
                      pending_to = peg;
                  }
              }
          }

          env.move.from = pending_from;
          env.move.to   = pending_to;

          if (pending_from < 0 || pending_to < 0) {
              c_render(&env);    // show partial selection text
              continue;          // wait until both pegs chosen
          }

          env.actions[0] = move_to_action(pending_from, pending_to, env.pegs);
          pending_from = pending_to = -1;
      } else {
          pending_from = pending_to = -1;
          env.actions[0] = rand() % (env.pegs * (env.pegs - 1));
      }

      c_step(&env);
      env.move.from = -1;
      env.move.to   = -1;
      c_render(&env);
  }


    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

