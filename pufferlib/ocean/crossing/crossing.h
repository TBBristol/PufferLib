#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"

/* CONSTS */

const unsigned char UNLOAD = 0;
const unsigned char LOAD = 1;
const unsigned char MOVE = 2;




#define OBS(env,r,c)  ((env)->observations[(r)*(env)->num_cols + (c)])
#define TYPE_COL        0 
#define ENTITY_ID_COL   (TYPE_COL + 3)
#define PAIR_ID_COL     (ENTITY_ID_COL + env->num_entities)
#define LOC_COL         (PAIR_ID_COL + env->num_entities)
#define BOAT_LOC_COL    (LOC_COL + 2)



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
   
    int num_cols; // timesaver
    int num_entities; // timesaver
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
/* -----------------------------------------------------------
 * Helpers
 * -----------------------------------------------------------*/

/* Return 1 iff row i is an “agent” (the odd-indexed member of a pair) */
static inline int is_agent(int i)               { return (i % 2) == 1; }

/* Return 1 iff row i is a “passenger” (the even-indexed member of a pair) */
static inline int is_passenger(int i)           { return (i % 2) == 0; }


/* True if entity i is currently at column col (bank or boat) */
static inline int at_col(RiverCrossing *env, int i, int col)
{
    return OBS(env, i, col) != 0;
}

static inline int in_a_boat(RiverCrossing *env, int i)
{
    return OBS(env, i, LOC_COL + 2);
}

/* -----------------------------------------------------------
 * 1 Count agents whose bit @loc_index == value
 * -----------------------------------------------------------*/
static int count_agents_where(RiverCrossing *env, int loc_index, unsigned char value)
{
    int cnt = 0;
    for (int i = 0; i < env->num_entities; ++i)
        if (is_agent(i) && OBS(env, i, loc_index) == value)
            ++cnt;
    return cnt;
}

static int count_passengers_where(RiverCrossing *env, int loc_index, unsigned char value)
{
    int cnt = 0;
    for (int i = 0; i < env->num_entities; ++i)
        if (is_passenger(i) && OBS(env, i, loc_index) == value)
            ++cnt;
    return cnt;
}

static int boat_col(RiverCrossing *env, int boat_idx)
{
    return BOAT_LOC_COL + boat_idx;
}

/* -----------------------------------------------------------
 * 2 Count agents currently inside boat <boat_idx>
 * -----------------------------------------------------------*/
static int count_agents_in_boat(RiverCrossing *env, unsigned char boat_idx)
{
    return count_agents_where(env, boat_col(env, boat_idx), 1);
}

static int count_passengers_in_boat(RiverCrossing *env, unsigned char boat_idx)
{
    return count_passengers_where(env, boat_col(env, boat_idx), 1);
}



/* -----------------------------------------------------------
 * 3 “Jealous-couple” safety rule:
 *    For every location (left bank, right bank, each boat),
 *    a passenger may not share that location with ANY agent
 *    other than their own agent, unless their own agent is
 *    also present.
 *
 *    Returns 1 if the configuration is safe, 0 if a conflict
 *    is detected.
 * -----------------------------------------------------------*/

static int passenger_agent_conflicts_ok(RiverCrossing *env)
{
    const int rows_pa = env->max_passengers * 2;   /* rows that hold (P,A) pairs */

    /* Loop over every passenger row (even index) */
    for (int p = 0; p < rows_pa; p += 2) {
        const int a = p + 1;                       /* that passenger’s own agent */

        /* --- Left bank ---------------------------------------------------- */
        if (OBS(env, p, LOC_COL) == 1) {
            const int own_here = OBS(env, a, LOC_COL);
            for (int k = 1; k < rows_pa; k += 2) { /* every agent row */
                if (k == a) continue;              /* skip own agent */
                if (OBS(env, k, LOC_COL) == 1 && !own_here)
                    return 0;                      /* foreign agent ⇒ conflict */
            }
        }

        /* --- Right bank --------------------------------------------------- */
        if (OBS(env, p, LOC_COL + 1) == 1) {
            const int own_here = OBS(env, a, LOC_COL + 1);
            for (int k = 1; k < rows_pa; k += 2) {
                if (k == a) continue;
                if (OBS(env, k, LOC_COL + 1) == 1 && !own_here)
                    return 0;
            }
        }

        /* --- Each boat ---------------------------------------------------- */
        for (int b = 0; b < env->max_boats; ++b) {
            const int boat_col_idx = BOAT_LOC_COL + b;
            if (OBS(env, p, boat_col_idx) == 1) {
                const int own_here = OBS(env, a, boat_col_idx);
                for (int k = 1; k < rows_pa; k += 2) {
                    if (k == a) continue;
                    if (OBS(env, k, boat_col_idx) == 1 && !own_here)
                        return 0;
                }
            }
        }
    }
    return 1;   /* no conflicts found */
}

// Required function RESET

void c_reset(RiverCrossing * env) {


    env->num_entities = env->max_passengers * 2 + env->max_boats;
    env->num_cols = 
                                3 // [type: agent, passenger, boat]
                                + env->num_entities        // ENTITY_ID_COL
                                + env->num_entities        // PAIR_ID_COL
                                + 2                        // LOC_COL (left/right bank)
                                + env->max_boats;          // extra (one for each boat)
    
    
     //clear obs
    memset(env->observations, 0, sizeof(int) * env->num_entities * env->num_cols);
    //set up obs [OHE entity type, OHE entity, OHE paired entity, OHE location]
    for (int i = 0; i < env->num_entities; ++i) {
    
    // Set ENTITY_ID one-hot
    OBS(env, i, ENTITY_ID_COL + i) = 1;
   
    // Set TYPE and PAIR_ID
    if (i < env->max_passengers * 2) {
        // Passengers are even rows (0, 2, 4...), agents are odd (1, 3, 5...)
        int is_passenger = (i % 2 == 0);
        if (is_passenger) {
            OBS(env, i, TYPE_COL + 0) = 1; // passenger
            OBS(env, i, PAIR_ID_COL + (i + 1)) = 1; // pair with agent below
          
        } else {
            OBS(env, i, TYPE_COL + 1) = 1; // agent
            OBS(env, i, PAIR_ID_COL + (i - 1)) = 1; // pair with passenger above
      
        }
    } else {
        OBS(env, i, TYPE_COL + 2) = 1; // boat
       
        // no pair
    }

    // All entities start on the left bank
    OBS(env, i, LOC_COL + 0) = 1;
}
  
   
    env->boat_capacity =2;
    env->tick = 0;
    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;
}

// Required Function Step

void c_step(RiverCrossing *env)
{
    //action comes in as [boat_idx, entity_idx, act_code]
    //act codes are unload, load, move
    int boat_index    = env->actions[0];
    int entity_index  = env->actions[1];
    int act_code      = env->actions[2];

    int boat_r        = env->max_passengers *2; // so with 3 passengers theres 3P 3A and then boats start on row 6 
  

    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;

    /* Termination shortcut  */
    #define FAIL(rew) do {env->rewards[0] = (rew); env->tick += 1; add_log(env); return; } while(0)

    if (act_code == MOVE) {
        /* boat must have at least one occupant */
        if (!count_agents_in_boat(env, (unsigned char)boat_index) && !count_passengers_in_boat(env, (unsigned char)boat_index) )  {
            FAIL(-1.0f);
        }
        /* move left→right or right→left */
        if (OBS(env, boat_r + boat_index, LOC_COL) == 1) {
            OBS(env, boat_r + boat_index, LOC_COL)  = 0;
            OBS(env, boat_r + boat_index , LOC_COL+1) = 1;


        } else if (OBS(env, boat_r + boat_index, LOC_COL + 1)== 1) {
            OBS(env, boat_r + boat_index, LOC_COL+ 1) = 0;
            OBS(env, boat_r + boat_index, LOC_COL)  = 1;
        } else {
            /* boat not on either bank? invalid */
            FAIL(-1.0f);
        }
        env->tick += 1;
    }
    else if (act_code == LOAD) {
        /* entity must share bank with boat and be boat‑less */
        if (OBS(env, entity_index, LOC_COL) != OBS(env, boat_r + boat_index, LOC_COL)) FAIL(-1.0f); //entity and boat not on same bank

        // add entity must not be in a boat condition

        if (
             OBS(env, entity_index, LOC_COL + 2) == 1 || OBS(env, entity_index, BOAT_LOC_COL + boat_index) == 1

        ) { 
            FAIL(-1.0f);
         }
        /* capacity test*/
        if (
            (
                count_agents_in_boat(env, (unsigned char) boat_index) +
                count_passengers_in_boat(env, (unsigned char) boat_index)
            ) 
            >= env->boat_capacity
        ) {
            FAIL(-1.0f);
        }


        /* load */

        OBS(env, entity_index, LOC_COL) = 0;
        OBS(env, entity_index, LOC_COL + 1) = 0;
        OBS(env, entity_index, LOC_COL + 2) = 1;
        OBS(env, entity_index, BOAT_LOC_COL + boat_index) = 1;

        
        env->rewards[0] = 0.2f; // loading gives small reward
        env->tick += 1;
    }


    else if (act_code == UNLOAD) {
        /* entity must currently be in that boat */
        if (
            OBS(env, entity_index, BOAT_LOC_COL + boat_index) != 1 &&
            OBS(env, entity_index, LOC_COL +2) != 1
        ) {
            FAIL(-1.0f);
        }

        /* unload to whichever side the boat is on */
        if (OBS(env, boat_r + boat_index, LOC_COL) == 1) {
            OBS(env, entity_index, LOC_COL)  = 1;
            OBS(env, entity_index, LOC_COL + 1) = 0;
           

        } else if (OBS(env, boat_r + boat_index, LOC_COL + 1) == 1) {
            OBS(env, entity_index, LOC_COL)  = 0;
            OBS(env, entity_index, LOC_COL + 1) = 1;

            env->rewards[0] = 0.2f; // unloading gives small reward but only if on right bank
        } else {
            FAIL(-1.0f);
        }
        OBS(env, entity_index,  BOAT_LOC_COL + boat_index) = 0;
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

    /* success if every passenger is on the right bank */
    int success = 1;
    for (int i = 0; i < env->max_passengers * 2; i += 2) {   // all passenger rows
        if (!OBS(env, i, LOC_COL+1)) {   // not on right bank
            success = 0;
            break;
        }
    }
    if (success) {
        env->terminals[0] = 1;
        env->rewards[0]   = 10.0f;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->tick >= env->max_ep_steps) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0f;
    add_log(env);
    c_reset(env);
    return;
}


    /* no termination; step continues */
    return;
}

/*--------------------------------------------------*
 *  Simple text-based render (raylib window)        *
 *--------------------------------------------------*/


/* one frame of rendering */
void c_render(RiverCrossing *env)
{
    /* -------------------------------------------------------------------- */
    /* 1.  Window init / teardown                                           */
    /* -------------------------------------------------------------------- */
    if (!IsWindowReady()) {                     /* first call only           */
        InitWindow(640, 480, "RiverCrossing (PufferLib)");
        SetTargetFPS(5);                        /* slow enough to watch      */
    }
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        return;
    }

    /* -------------------------------------------------------------------- */
    /* 2.  Draw the scene                                                   */
    /* -------------------------------------------------------------------- */
    BeginDrawing();
    ClearBackground((Color){24, 24, 24, 255});

    /* Bank headings */
    DrawText("Left Bank",  40, 40, 20, (Color){200,200,200,255});
    DrawText("River",     280, 40, 20, (Color){200,200,200,255});
    DrawText("Right Bank",480, 40, 20, (Color){200,200,200,255});

    int y  = 80;          /* top row for entities */
    int dy = 20;          /* vertical spacing     */

    int total_entities = env->max_passengers * 2 + env->boats;

    for (int i = 0; i < total_entities; ++i) {
        /* Build a label --------------------------------------------------- */
        char buf[64];
        if (i < env->max_passengers * 2) {
            if ((i & 1) == 0) {      /* even rows → passenger */
                sprintf(buf, "Passenger %d", i / 2);
            } else {                 /* odd  rows → agent     */
                sprintf(buf, "Agent %d",     i / 2);
            }
        } else {                     /* boats are after all P/A rows */
            sprintf(buf, "Boat %d", i - env->max_passengers * 2);
        }

        /* Where is this entity? ------------------------------------------ */
        int x;
        if (OBS(env, i, LOC_COL) == 1)          x = 40;   /* left bank  */
        else if (OBS(env, i, LOC_COL + 1) == 1) x = 480;  /* right bank */
        else                                    x = 280;  /* in a boat  */

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