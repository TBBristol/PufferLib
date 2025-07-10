/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "crossing.h"

int main() {
    RiverCrossing env = {.boats = 1, 
                        .passengers = 2, 
                        .max_passengers = 2,
                        .max_boats = 1, };
                        
    int num_rows = 1 + env.max_passengers * 2 + env.max_boats;
    env.observations = (unsigned char*)calloc(num_rows * 3, sizeof(unsigned char));
    env.actions = (int*)calloc(3, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    for (int frame = 0; frame < 60 && !WindowShouldClose(); ++frame) {
    c_render(&env);
}
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

