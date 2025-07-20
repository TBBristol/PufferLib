/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "stacking.h"

int main() {
    ContainerStacking env = {.num_containers = 10, 
                            .max_ep_steps = 100};
                        
   
   // env.observations = (int*)calloc(env.num_entities * env.num_cols, sizeof(int));
    //env.actions = (int*)calloc(3, sizeof(int));
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

