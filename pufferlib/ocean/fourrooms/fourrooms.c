#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fourrooms.h"

static void alloc_buffers(Fourrooms* env) {
    const size_t obs_count = (size_t)FOURROOMS_OBS_LAYERS * (size_t)env->size * (size_t)env->size;
    env->observations = calloc(obs_count, sizeof(unsigned char));
    env->actions = calloc(1, sizeof(int));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(unsigned char));
}

static void free_buffers(Fourrooms* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    env->observations = NULL;
    env->actions = NULL;
    env->rewards = NULL;
    env->terminals = NULL;
}

int main(void) {
    Fourrooms env = {
        .size = 10,
        .max_steps = 200,
        .client = NULL,
    };

    srand((unsigned int)time(NULL));
    alloc_buffers(&env);
    init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        int action = NOOP;
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            action = UP;
        } else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
            action = DOWN;
        } else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
            action = LEFT;
        } else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            action = RIGHT;
        }

        env.actions[0] = action;
        if (action != NOOP) {
            c_step(&env);
        }
        c_render(&env);
    }

    c_close(&env);
    free_buffers(&env);
    return 0;
}
