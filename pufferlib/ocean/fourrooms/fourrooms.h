#ifndef PUFFERLIB_OCEAN_FOURROOMS_H
#define PUFFERLIB_OCEAN_FOURROOMS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define FOURROOMS_OBS_LAYERS 3
#define TILE 32

enum {
    NOOP = 0,
    DOWN = 1,
    UP = 2,
    LEFT = 3,
    RIGHT = 4,
};

enum {
    AGENT = 0,
    WALLS = 1,
    TARGET = 2,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float on_targets;
    float n;
} Log;

typedef struct {
    Texture2D wall;
    Texture2D target;
    Texture2D floor;
    Texture2D agent;
} Client;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    int max_steps;
    int agent_x;
    int agent_y;
    Client* client;
    int win;
} Fourrooms;

static inline int obs_stride(const Fourrooms* env) {
    return env->size * env->size;
}

static inline bool in_bounds(const Fourrooms* env, int x, int y) {
    return x >= 0 && y >= 0 && x < env->size && y < env->size;
}

static inline void set_entity(Fourrooms* env, int entity, int x, int y, unsigned char value) {
    env->observations[entity * obs_stride(env) + y * env->size + x] = value;
}

static inline unsigned char get_entity(const Fourrooms* env, int entity, int x, int y) {
    return env->observations[entity * obs_stride(env) + y * env->size + x];
}

static inline void clear_dynamic_layers(Fourrooms* env) {
    const size_t layer_size = (size_t)obs_stride(env);
    memset(env->observations + AGENT * layer_size, 0, layer_size * sizeof(unsigned char));
    memset(env->observations + TARGET * layer_size, 0, layer_size * sizeof(unsigned char));
}

static inline uint32_t get_random_open_idx(const Fourrooms* env, bool avoid_agent, bool avoid_target) {
    if (env->size <= 0) {
        return UINT32_MAX;
    }

    const uint32_t total = (uint32_t)obs_stride(env);
    const uint32_t start = (uint32_t)(rand() % (int)total);

    for (uint32_t offset = 0; offset < total; offset++) {
        const uint32_t idx = (start + offset) % total;
        const int x = (int)(idx % (uint32_t)env->size);
        const int y = (int)(idx / (uint32_t)env->size);
        if (get_entity(env, WALLS, x, y) != 0) {
            continue;
        }
        if (avoid_agent && get_entity(env, AGENT, x, y) != 0) {
            continue;
        }
        if (avoid_target && get_entity(env, TARGET, x, y) != 0) {
            continue;
        }
        return idx;
    }

    return UINT32_MAX;
}

static inline void build_map(Fourrooms* env) {
    const size_t total_obs = (size_t)FOURROOMS_OBS_LAYERS * (size_t)obs_stride(env);
    memset(env->observations, 0, total_obs * sizeof(unsigned char));

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (x == 0 || y == 0 || x == env->size - 1 || y == env->size - 1) {
                set_entity(env, WALLS, x, y, 1);
            }
        }
    }

    if (env->size < 5) {
        return;
    }

    const int divide_idx = env->size / 2;
    const int door_a = env->size / 4;
    const int door_b = env->size - door_a - 1;

    for (int x = 1; x < env->size - 1; x++) {
        set_entity(env, WALLS, x, divide_idx, 1);
    }
    for (int y = 1; y < env->size - 1; y++) {
        set_entity(env, WALLS, divide_idx, y, 1);
    }

    set_entity(env, WALLS, divide_idx, door_a, 0);
    set_entity(env, WALLS, divide_idx, door_b, 0);
    set_entity(env, WALLS, door_a, divide_idx, 0);
    set_entity(env, WALLS, door_b, divide_idx, 0);
}

static inline void init(Fourrooms* env) {
    env->tick = 0;
    env->win = 0;
    env->agent_x = 0;
    env->agent_y = 0;
    build_map(env);
}

static inline void add_log(Fourrooms* env) {
    const float won = env->win ? 1.0f : 0.0f;
    env->log.perf += won;
    env->log.score += env->rewards[0];
    env->log.episode_length += (float)env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.on_targets += won;
    env->log.n += 1.0f;
}

static inline void c_reset(Fourrooms* env) {
    clear_dynamic_layers(env);

    const uint32_t goal_idx = get_random_open_idx(env, false, false);
    if (goal_idx == UINT32_MAX) {
        env->tick = 0;
        env->win = 0;
        return;
    }

    const int goal_x = (int)(goal_idx % (uint32_t)env->size);
    const int goal_y = (int)(goal_idx / (uint32_t)env->size);
    set_entity(env, TARGET, goal_x, goal_y, 1);

    const uint32_t agent_idx = get_random_open_idx(env, false, true);
    if (agent_idx == UINT32_MAX) {
        set_entity(env, TARGET, goal_x, goal_y, 0);
        env->tick = 0;
        env->win = 0;
        return;
    }

    env->agent_x = (int)(agent_idx % (uint32_t)env->size);
    env->agent_y = (int)(agent_idx / (uint32_t)env->size);
    set_entity(env, AGENT, env->agent_x, env->agent_y, 1);
    env->tick = 0;
    env->win = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

static inline void move_entity(Fourrooms* env, unsigned char entity, int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

static inline bool clear(const Fourrooms* env, int x, int y) {
    if (!in_bounds(env, x, y)) {
        return false;
    }
    return get_entity(env, WALLS, x, y) == 0;
}

static inline bool take_action(Fourrooms* env, int action) {
    int dx = 0;
    int dy = 0;

    if (action == DOWN) {
        dy = 1;
    } else if (action == UP) {
        dy = -1;
    } else if (action == LEFT) {
        dx = -1;
    } else if (action == RIGHT) {
        dx = 1;
    } else {
        return false;
    }

    const int next_x = env->agent_x + dx;
    const int next_y = env->agent_y + dy;
    if (!in_bounds(env, next_x, next_y)) {
        return false;
    }

    if (get_entity(env, TARGET, next_x, next_y) == 1) {
        move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
        env->agent_x = next_x;
        env->agent_y = next_y;
        return true;
    }

    if (clear(env, next_x, next_y)) {
        move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
        env->agent_x = next_x;
        env->agent_y = next_y;
    }

    return false;
}

static inline void c_step(Fourrooms* env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    const bool goal = take_action(env, env->actions[0]);
    if (goal) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        env->win = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        env->win = 0;
        add_log(env);
        c_reset(env);
    }
}

static inline Client* c_create(Fourrooms* env) {
    Client* client = calloc(1, sizeof(Client));
    const char* sprite_search_paths[] = {
        "sprites_pack/PNG",
        "pufferlib/ocean/fourrooms/sprites_pack/PNG",
        "../pufferlib/ocean/fourrooms/sprites_pack/PNG",
    };
    const char* sprite_base = NULL;

    for (unsigned int i = 0; i < sizeof(sprite_search_paths) / sizeof(sprite_search_paths[0]); i++) {
        if (DirectoryExists(sprite_search_paths[i])) {
            sprite_base = sprite_search_paths[i];
            break;
        }
    }

    if (sprite_base == NULL) {
        TraceLog(LOG_WARNING, "Fourrooms sprites not found next to executable, using default relative path");
        sprite_base = "sprites_pack/PNG";
    }

    char resource_path[256];

    snprintf(resource_path, sizeof(resource_path), "%s/Wall_Black.png", sprite_base);
    client->wall = LoadTexture(resource_path);
    snprintf(resource_path, sizeof(resource_path), "%s/EndPoint_Black.png", sprite_base);
    client->target = LoadTexture(resource_path);
    snprintf(resource_path, sizeof(resource_path), "%s/GroundGravel_Concrete.png", sprite_base);
    client->floor = LoadTexture(resource_path);
    client->agent = LoadTexture("resources/shared/puffers_128.png");

    env->client = client;
    return client;
}

static inline void draw_tile(Fourrooms* env, int x, int y) {
    Client* c = env->client;
    Rectangle dest = {(float)(x * TILE), (float)(y * TILE), (float)TILE, (float)TILE};

    DrawTexturePro(
        c->floor,
        (Rectangle){0, 0, (float)c->floor.width, (float)c->floor.height},
        dest,
        (Vector2){0, 0},
        0.0f,
        WHITE);

    if (get_entity(env, TARGET, x, y)) {
        DrawTexturePro(
            c->target,
            (Rectangle){0, 0, (float)c->target.width, (float)c->target.height},
            dest,
            (Vector2){0, 0},
            0.0f,
            WHITE);
    }

    if (get_entity(env, WALLS, x, y)) {
        DrawTexturePro(
            c->wall,
            (Rectangle){0, 0, (float)c->wall.width, (float)c->wall.height},
            dest,
            (Vector2){0, 0},
            0.0f,
            WHITE);
    }

    if (get_entity(env, AGENT, x, y)) {
        Rectangle src = {0, 0, c->agent.width / 2.0f, (float)c->agent.height};
        DrawTexturePro(c->agent, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
    }
}

static inline void c_render(Fourrooms* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE * env->size, TILE * env->size, "PufferLib Fourrooms");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = c_create(env);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            draw_tile(env, x, y);
        }
    }

    EndDrawing();
}

static inline void c_close(Fourrooms* env) {
    if (IsWindowReady()) {
        if (env->client != NULL) {
            UnloadTexture(env->client->wall);
            UnloadTexture(env->client->target);
            UnloadTexture(env->client->floor);
            UnloadTexture(env->client->agent);
            free(env->client);
            env->client = NULL;
        }
        CloseWindow();
    }
}

#endif
