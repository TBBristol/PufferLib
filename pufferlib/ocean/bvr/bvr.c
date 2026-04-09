// Standalone C demo for bvr environment
// Compile using: ./scripts/build_ocean.sh bvr [local|fast]
// Run with: ./bvr

#include "bvr.h"
#include "render.h"
#include <stdio.h>
#include <time.h>

void generate_dummy_actions(BvrEnv *env) {
    // Generate random floats in [-1, 1] range
    env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

void test_performance(int timeout) {
    srand(time(NULL)); // Seed random number generator

    BvrEnv *env = calloc(1, sizeof(BvrEnv));
    env->num_agents = 1;
    init(env);

    size_t obs_size = OBS_DIM * env->framestack;
    size_t act_size = 3;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(float));

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return;
    }

    init(env);
    c_reset(env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        generate_dummy_actions(env);
        c_step(env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = (env->num_agents * num_steps) / (float)(end - start);
    printf("Test Environment SPS: %f\n", sps);

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
}

int main() {
    srand(time(NULL)); // Seed random number generator

    BvrEnv *env = calloc(1, sizeof(BvrEnv));
    env->num_agents = 1;
    init(env);

    size_t obs_size = OBS_DIM * env->framestack;
    size_t act_size = 3;
    env->observations = (float *)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float *)calloc(env->num_agents * act_size, sizeof(float));
    env->rewards = (float *)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->num_agents, sizeof(float));

    if (!env->observations || !env->actions || !env->rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env);
        return 0;
    }

    init(env);
    c_reset(env);
    c_render(env);

    int action_hold = 20;
    generate_dummy_actions(env);
    while (!WindowShouldClose()) {
        if (env->tick % action_hold == 0) {
            generate_dummy_actions(env);
        }
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);

    return 0;
}
