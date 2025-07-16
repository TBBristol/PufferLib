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
                        .passengers = 1, 
                        .max_passengers = 1,
                        .max_boats = 1, };
                        
    env.num_entities = env.max_passengers * 2 + env.max_boats;
    env.num_cols = 
                                3 // [type: agent, passenger, boat]
                                + env.num_entities        // ENTITY_ID_COL
                                + env.num_entities        // PAIR_ID_COL
                                + 2                        // LOC_COL (left/right bank)
                                + env.max_boats;          // extra (one for each boat)
    env.observations = (int*)calloc(env.num_entities * env.num_cols, sizeof(int));
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

