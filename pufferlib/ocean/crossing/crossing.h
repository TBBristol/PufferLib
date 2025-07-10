#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"


/* CONSTS */

const unsigned char UNLOAD = 0;
const unsigned char LOAD = 1;
const unsigned char MOVE = 2;

#define LEFT_COL  0
#define BOAT_COL   1
#define RIGHT_COL  2
#define OBS(r,c)   env->observations[(r)*3 + (c)]



// Required struct by pufferlib. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    int* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int tick;
    // params
    int passengers;
    int boats;
    int max_passengers;
    int max_boats;
    int boat_capacity; //default 2 set in reset
    int num_rows; // timesaver
    int max_ep_steps;
} RiverCrossing;

void add_log(RiverCrossing* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

/*--------------------------------------------------*
 *  Internal helpers                                  *
 *--------------------------------------------------*/
static inline int boat_row(RiverCrossing *env, int boat_index)
{   /* boat_index is 0‑based */
    return 1 + env->max_passengers * 2 + boat_index;
}

/* count agents that satisfy a boolean predicate on location columns */
static int count_agents_where(RiverCrossing *env, int col, unsigned char value)
{
    int count = 0;
    int first_agent_row = 1 + env->passengers;
    int last_agent_row  = first_agent_row + env->passengers; /* exclusive */
    for (int r = first_agent_row; r < last_agent_row; ++r) {
        if (OBS(r, col) == value)
            ++count;
    }
    return count;
}

static int count_agents_in_boat(RiverCrossing *env, unsigned char boat_id)
{
    int count = 0;
    int first_agent_row = 1 + env->passengers;
    int last_agent_row  = first_agent_row + env->passengers;
    for (int r = first_agent_row; r < last_agent_row; ++r) {
        if (OBS(r, BOAT_COL) == boat_id)
            ++count;
    }
    return count;
}

static int passenger_agent_conflicts_ok(RiverCrossing *env)
{
    /* Returns 1 when the invariant holds (i.e. no conflict), 0 otherwise */
    for (int i = 0; i < env->passengers; ++i) {
        int passenger_row = 1 + i;
        int agent_row     = 1 + env->passengers + i;

        /* locate passenger */
        int passenger_left  = OBS(passenger_row, LEFT_COL);
        int passenger_right = OBS(passenger_row, RIGHT_COL);
        unsigned char passenger_boat = OBS(passenger_row, BOAT_COL);

        /* locate agent */
        int agent_left  = OBS(agent_row, LEFT_COL);
        int agent_right = OBS(agent_row, RIGHT_COL);
        unsigned char agent_boat = OBS(agent_row, BOAT_COL);

        /* same place? then fine */
        if ((passenger_left  && agent_left)  ||
            (passenger_right && agent_right) ||
            (passenger_boat  && passenger_boat == agent_boat))
            continue;

        /* otherwise passenger must be alone (no other agents) where they stand */
        if (passenger_left) {
            if (count_agents_where(env, LEFT_COL, 1) != 0)
                return 0;
        } else if (passenger_right) {
            if (count_agents_where(env, RIGHT_COL, 1) != 0)
                return 0;
        } else { /* in a boat */
            if (count_agents_in_boat(env, passenger_boat) != 0)
                return 0;
        }
    }
    return 1;
}

// Required function RESET

void c_reset(RiverCrossing * env) {

    env->num_rows = 1 + env->max_passengers * 2 + env->max_boats;
    //clear obs
    memset(env->observations, 0, sizeof(int) * env->num_rows * 3);
    /* header */
    OBS(0, 0) = (unsigned char)env->passengers;
    OBS(0, 1) = (unsigned char)env->passengers; 
    OBS(0, 2) = (unsigned char)env->boats;
    
    /* place all entities on the left bank */
    int last_entity_row = 1 + env->passengers * 2 + env->boats;
    for (int r = 1; r < last_entity_row; ++r) {
        OBS(r, LEFT_COL) = 1;
    }
    env->boat_capacity =2;
    env->tick = 0;
}



// Required Function Step

void c_step(RiverCrossing *env)
{
    
    int boat_index    = env->actions[0];
    int entity_index  = env->actions[1];
    int act_code      = env->actions[2];

   
    int boat_id       = boat_index + 1;        /* boat IDs start at 1 */
    int boat_r        = boat_row(env, boat_index);
    int entity_r      = 1 + entity_index;      /* passengers first, then agents */

    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;

    /* Termination shortcut  */
    #define FAIL(rew) do { env->terminals[0] = 1; env->rewards[0] = (rew); add_log(env); c_reset(env); return; } while(0)

    if (act_code == MOVE) {
        /* boat must have at least one occupant */
        if (!count_agents_in_boat(env, (unsigned char)boat_id)) {
            FAIL(-1.0f);
        }
        /* move left→right or right→left */
        if (OBS(boat_r, LEFT_COL)) {
            OBS(boat_r, LEFT_COL)  = 0;
            OBS(boat_r, RIGHT_COL) = 1;
        } else if (OBS(boat_r, RIGHT_COL)) {
            OBS(boat_r, RIGHT_COL) = 0;
            OBS(boat_r, LEFT_COL)  = 1;
        } else {
            /* boat not on either bank? invalid */
            FAIL(-1.0f);
        }
        env->tick += 1;
    }
    else if (act_code == LOAD) {
        /* entity must share bank with boat and be boat‑less */
        if (OBS(entity_r, BOAT_COL) != 0) FAIL(-1.0f);
        if (OBS(boat_r, LEFT_COL) != OBS(entity_r, LEFT_COL)) FAIL(-1.0f);
        if (OBS(boat_r, RIGHT_COL) != OBS(entity_r, RIGHT_COL)) FAIL(-1.0f);
        /* capacity */
        if (count_agents_in_boat(env, (unsigned char)boat_id) >= env->boat_capacity) FAIL(-1.0f);
        /* load */
        OBS(entity_r, LEFT_COL)  = 0;
        OBS(entity_r, RIGHT_COL) = 0;
        OBS(entity_r, BOAT_COL)  = (unsigned char)boat_id;
        env->tick += 1;
    }
    else if (act_code == UNLOAD) {
        /* entity must currently be in that boat */
        if (OBS(entity_r, BOAT_COL) != (unsigned char)boat_id) FAIL(-1.0f);
        /* unload to whichever side the boat is on */
        if (OBS(boat_r, LEFT_COL)) {
            OBS(entity_r, LEFT_COL)  = 1;
            OBS(entity_r, RIGHT_COL) = 0;
        } else if (OBS(boat_r, RIGHT_COL)) {
            OBS(entity_r, LEFT_COL)  = 0;
            OBS(entity_r, RIGHT_COL) = 1;
        } else {
            FAIL(-1.0f);
        }
        OBS(entity_r, BOAT_COL) = 0;
        env->tick += 1;
    }
    else {
        FAIL(-1.0f); /* invalid action code */
    }

    /* ----------------------------------------------------------------
     *  Post‑step checks                                               */
    /* passenger‑agent pairing rule */
    if (!passenger_agent_conflicts_ok(env)) {
        env->terminals[0] = 1;
        env->rewards[0]   = -1.0f;
        add_log(env);
        c_reset(env);
        return;
    }

    /* success if every passenger & agent is on the right bank */
    int success = 1;
    int last_entity_row = 1 + env->passengers * 2;
    for (int r = 1; r < last_entity_row; ++r) {
        if (!OBS(r, RIGHT_COL)) { success = 0; break; }
    }
    if (success) {
        env->terminals[0] = 1;
        env->rewards[0]   = 1.0f;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->tick >= env->max_ep_steps) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0f; // or negative if you want
    add_log(env);
    c_reset(env);
    return;
}


    /* no termination; step continues */
    return;
}


/*--------------------------------------------------*
 *  Simple text‑based render (raylib window)          *
 *--------------------------------------------------*/
void c_render(RiverCrossing *env)
{
    if (!IsWindowReady()) {
        InitWindow(640, 480, "RiverCrossing (PufferLib)");
        SetTargetFPS(5);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){24, 24, 24, 255});

    /* draw banks */
    DrawText("Left Bank", 40, 40, 20, (Color){200,200,200,255});
    DrawText("River",     280,40, 20, (Color){200,200,200,255});
    DrawText("Right Bank",480,40, 20, (Color){200,200,200,255});

    int y = 80;
    int dy = 20;

    int total_entities = env->passengers * 2 + env->boats;
    for (int i = 0; i < total_entities; ++i) {
        int r = 1 + i;
        char buf[64];
        if (i < env->passengers) {
            sprintf(buf, "Passenger %d", i);
        } else if (i < env->passengers*2) {
            sprintf(buf, "Agent %d", i - env->passengers);
        } else {
            sprintf(buf, "Boat %d", i - env->passengers*2 + 1);
        }
        int x = 0;
        if (OBS(r, LEFT_COL))      x = 40;
        else if (OBS(r, RIGHT_COL)) x = 480;
        else                       x = 280;
        DrawText(buf, x, y, 16, (Color){0, 187, 187, 255});
        y += dy;
    }

    EndDrawing();
}

/*--------------------------------------------------*
 *  Close                                             *
 *--------------------------------------------------*/
void c_close(RiverCrossing *env)
{
    if (IsWindowReady()) {
        CloseWindow();
    }
}