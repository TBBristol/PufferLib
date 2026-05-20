#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "raylib.h"
#include "boxoban_maps.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char AGENT = 0;
const unsigned char WALLS = 1;
const unsigned char BOXES = 2;
const unsigned char TARGET = 3;

#define BOXOBAN_SIZE 10
#define BOXOBAN_OBS_SIZE (4*BOXOBAN_SIZE*BOXOBAN_SIZE)

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float on_targets; // Number of targets currently boxed
    float n; // Required as the last field 
} Log;

typedef struct {
    Texture2D wall;
    Texture2D box;
    Texture2D target;
    Texture2D floor;
    Texture2D agent;
    Texture2D box_on_target;
} Client;

typedef struct {
    int tick;
    int agent_x;
    int agent_y;
    bool initialized;
    unsigned char observations[BOXOBAN_OBS_SIZE];
    unsigned char intermediate_rewards[BOXOBAN_SIZE*BOXOBAN_SIZE];
    int on_target; //num targets currently boxed
    int n_boxes; //boxes in map
    int n_targets; //targets in map
    int win;
    float episode_return;
} State;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    float* terminals; // Required. We don't yet have truncations as standard yet
    State state;
    unsigned int rng;
    int size;
    int num_agents;
    int max_steps;
    float int_r_coeff;
    float target_loss_pen_coeff;
    int difficulty_id; // 0=basic,1=easy,2=medium,3=hard,4=unfiltered
    Client* client;
} Boxoban;

void ensure_map_loaded(void);

static int boxoban_configure_maps_from_env(Boxoban* env) {
    if (env->difficulty_id == -1) {
        return 0;
    }

    if (env->difficulty_id < -1) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }

    const char* difficulty_name = boxoban_difficulty_name_from_id(env->difficulty_id);
    if (difficulty_name == NULL) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }
    char prepared_path[512];
    if (boxoban_prepare_maps_for_difficulty(difficulty_name, prepared_path, sizeof(prepared_path)) != 0) {
        return -1;
    }

    return 0;
}

//Entity,x,y  convention y moves top to bottom

static inline void set_entity(Boxoban *env, int entity, int x, int y, unsigned char value) {
    int idx = (entity)*env->size*env->size + (y)*env->size + (x);
    env->state.observations[idx] = value;
    if (env->observations) {
        env->observations[idx] = value;
    }
}

static inline unsigned char get_entity(Boxoban *env, int entity, int x, int y) {
    return env->state.observations[(entity)*env->size*env->size + (y)*env->size + (x)];
}

static inline void set_intermediate_reward(Boxoban *env, int x, int y, unsigned char value) {
    env->state.intermediate_rewards[(y)*env->size + (x)] = value;
}

static inline unsigned char get_intermediate_reward_status(Boxoban *env, int x, int y) {
    return env->state.intermediate_rewards[(y)*env->size + (x)];
}

static inline uint32_t get_random_puzzle_idx(Boxoban *env) {
    int idx = rand_r(&env->rng) % PUZZLE_COUNT;
    return idx;
}


void init (Boxoban* env) {
    static int boxoban_maps_ready = 0;
    if (!boxoban_maps_ready) {
        if (boxoban_configure_maps_from_env(env) != 0) {
            fprintf(stderr, "Failed to configure Boxoban maps\n");
            abort();
        }
        ensure_map_loaded();
        boxoban_maps_ready = 1;
    }
    env->state.win = 0;
    env->state.initialized = false;
  }


void add_log(Boxoban* env) {
    float perf = (env->state.win== 1) ? 1.0f : 0.0f;
    env->log.perf += perf;
    env->log.score += perf;
    env->log.episode_length += env->state.tick;
    env->log.episode_return += env->state.episode_return;
    env->log.on_targets += env->state.on_target;
    env->log.n++;
}


bool clear(Boxoban* env, int x, int y) {
    if (x < 0 || y < 0 || x >= env->size || y >= env->size) {
        return false;
    }
    return (get_entity(env, WALLS, x, y) == 0) && (get_entity(env, BOXES, x, y) == 0);
}

void refresh_state(Boxoban* env) {
    if (env->observations) {
        memcpy(env->observations, env->state.observations, BOXOBAN_OBS_SIZE);
    }
}

// Required function
void c_reset(Boxoban* env) {
    const uint32_t i = get_random_puzzle_idx(env);
    const uint8_t* puzzle = MAP_BASE + (size_t)i * PUZZLE_SIZE;
    memcpy(env->state.observations, puzzle, PUZZLE_OBS_BYTES);

    const uint8_t* meta = puzzle + PUZZLE_OBS_BYTES;
    env->state.agent_x = (int)meta[0];
    env->state.agent_y = (int)meta[1];
    env->state.n_boxes = (int)meta[2];
    env->state.n_targets = (int)meta[3];
    env->state.on_target = (int)meta[4];

    memcpy(env->state.intermediate_rewards,
            env->state.observations + TARGET * env->size * env->size,env->size * env->size);

    env->state.tick = 0;
    env->state.win = 0;
    env->state.episode_return = 0;

    if (!env->state.initialized) {
        env->state.tick = rand_r(&env->rng) % env->max_steps;
        env->state.initialized = true;
    }
    refresh_state(env);
}

//Updates OBS for moved entity
void move_entity(Boxoban* env,unsigned char entity,int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

//Updates state and intermediate reward array in place
int take_action(Boxoban* env, int action) {

    int dx = 0;
    int dy = 0;
    int int_r = 0;

    if (action == NOOP) {
        return 0;
    }
    else if (action == DOWN) {
        dy = 1;
    }
    else if (action == UP) {
        dy = -1;
    }
    else if (action == LEFT) {
        dx = -1;
    }
    else if (action == RIGHT) {
        dx = 1;
    }

    //if move space is clear, move agent
    if (clear(env, env->state.agent_x + dx, env->state.agent_y + dy)) {
        
        move_entity(env, AGENT, env->state.agent_x, env->state.agent_y, dx, dy);
        env->state.agent_y += dy;
        env->state.agent_x += dx;
        return 0;
    }
    //if its not clear, but its a box and box is clear to move, move both
    else if (clear(env, env->state.agent_x+ 2*dx, env->state.agent_y + 2*dy)
            && get_entity(env, BOXES, env->state.agent_x + dx, env->state.agent_y + dy) == 1) {

            //if box is on target currently, remove from on_target count
            if (get_entity(env, TARGET, env->state.agent_x + dx, env->state.agent_y + dy) == 1) {

                env->state.on_target -= 1;
            }
            //move both entities
            move_entity(env, BOXES, env->state.agent_x + dx, env->state.agent_y + dy, dx, dy);
            move_entity(env, AGENT, env->state.agent_x, env->state.agent_y, dx, dy);
            env->state.agent_y += dy;
            env->state.agent_x += dx;
        
            //if box is now on target, add to on_target count
            //if its a new target recieve intermediate reward and zero out intermediate reward
            if (get_entity(env, TARGET, env->state.agent_x + dx, env->state.agent_y + dy) == 1) {
                
                env->state.on_target += 1;
                int_r = get_intermediate_reward_status(env, env->state.agent_x + dx, env->state.agent_y + dy);
                set_intermediate_reward(env, env->state.agent_x + dx, env->state.agent_y + dy, 0);
            }
            return int_r;
    }
    return 0;
}

// Required function
void c_step(Boxoban* env) {
    env->state.tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
       
    int action = (int)env->actions[0];

    float on_target = env->state.on_target;
    int int_r = take_action(env, action); //int_r _new_ tgts covered, modifies observations in place
    float on_target_after = env->state.on_target;
                                          
    env->rewards[0] += (float)int_r * env->int_r_coeff; //coeff in .ini
 
    if (on_target_after < on_target) { //target loss penalty
        env->rewards[0] -= env->target_loss_pen_coeff; //coeff in .ini
    }

    //Terminals
    if (env->state.on_target == env->state.n_targets) {
        env->terminals[0] = 1;
        env->rewards[0] += 1.0;
        env->state.win = 1;
        env->state.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->state.tick >= env->max_steps) {
        env->terminals[0] = 1;
        env->rewards[0] -= 1.0; 
        env->state.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    env->state.episode_return += env->rewards[0];

}

Client* c_create(Boxoban* env) {
    Client* client = (Client*)calloc(1,sizeof(Client));
    client->wall = LoadTexture("resources/boxoban/Wall_Black.jpg");
    client->box = LoadTexture("resources/boxoban/Crate_Black.jpg");
    client->target = LoadTexture("resources/boxoban/EndPoint_Black.jpg");
    client->floor = LoadTexture("resources/boxoban/GroundGravel_Concrete.jpg");
    client->box_on_target = LoadTexture("resources/boxoban/EndPoint_Blue.jpg");
    client->agent = LoadTexture("resources/shared/puffers_128.png");
    env-> client = client;
    return client;
}

#define TILE 32

Texture2D choose_sprite(Client *c, Boxoban *env, int x, int y) {
    int a = get_entity(env, AGENT, x, y);
    int w = get_entity(env, WALLS, x, y);
    int b = get_entity(env, BOXES, x, y);
    int t = get_entity(env, TARGET, x, y);

    if (w) return c->wall;
    if (b && t) return c->box_on_target;
    if (b) return c->box;
    if (a) return c->agent;
    if (t) return c->target;

    return c->floor;
}

void draw_tile(Boxoban *env, int x, int y) {
      Client *c = env->client;
      Rectangle dest = {x * TILE, y * TILE, TILE, TILE};

      // Always lay down the base tile
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
      if (get_entity(env, BOXES, x, y)) {
          Texture2D tex = get_entity(env, TARGET, x, y) ? c->box_on_target : c->box;
          DrawTexturePro(
              tex,
              (Rectangle){0, 0, (float)tex.width, (float)tex.height},
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


// Required function. Should handle creating the client on first call
void c_render(Boxoban* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE*env->size, TILE*env->size, "PufferLib Boxoban");
        SetTargetFPS(10);
    }

    // Standard across our envs so exiting is always the same
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

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Boxoban* env) {
    if (IsWindowReady()) {
        if (env->client) {
            UnloadTexture(env->client->wall);
            UnloadTexture(env->client->box);
            UnloadTexture(env->client->target);
            UnloadTexture(env->client->floor);
            UnloadTexture(env->client->agent);
            free(env->client);
            env->client = NULL;
        }
        CloseWindow();
    }
}
