/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "stacking.h"

int main() {
    ContainerStacking env = {.num_containers = 30, 
                            .max_ep_steps = 100,
                                      .max_height = 4,
                                    .num_stacks = 8,
                                        .reward_max_breach = -23,
                                        .reset_max_breach =1};
   
    env.observations = (float*)calloc(env.num_stacks * env.max_height, sizeof(float));
    env.actions = (int*)calloc(8, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    
    
    int acts = 5;

    c_reset(&env);
    //c_render(&env);
    while (acts >= 0){
    env.actions[0] = rand() % 8;
    c_step(&env);

    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

