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

/* -----------------------------------------------------------
 * 1 Count agents whose bit @loc_index == value
 * -----------------------------------------------------------*/
static int count_agents_where(RiverCrossing *env, int loc_index, unsigned char value)
{
    int cnt = 0;
    for (int i = 0; i < env->passengers * 2; ++i)
        if (is_agent(i) && OBS(env, i, loc_index) == value)
            ++cnt;
    return cnt;
}

static int count_passengers_where(RiverCrossing *env, int loc_index, unsigned char value)
{
    int cnt = 0;
    for (int i = 0; i < env-> passengers *2; ++i)
        if (is_passenger(i) && OBS(env, i, loc_index) == value)
            ++cnt;
    return cnt;
}

static int boat_col(RiverCrossing *env, int boat_idx)
{
    return BOAT_LOC_COL + boat_idx;
}

static int in_a_boat(RiverCrossing *env, int entity_idx)
{
    for (int b = 0; b < env->boats; ++b) {
        if (OBS(env, entity_idx, BOAT_LOC_COL + b) == 1)
            return 1;
    }
    return 0;
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
    const int rows_pa = env->passengers * 2;   /* rows that hold (P,A) pairs */

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
        for (int b = 0; b < env->boats; ++b) {
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
    
   
   
    // Set TYPE and PAIR_ID
    if (i < env->passengers * 2) {
        // Passengers are even rows (0, 2, 4...), agents are odd (1, 3, 5...)
        int is_passenger = (i % 2 == 0);
        if (is_passenger) {
            OBS(env, i, TYPE_COL + 0) = 1; // passenger
            OBS(env, i, PAIR_ID_COL + (i + 1)) = 1; // pair with agent below
          
        } else {
            OBS(env, i, TYPE_COL + 1) = 1; // agent
            OBS(env, i, PAIR_ID_COL + (i - 1)) = 1; // pair with passenger above
      
        }
        // All entities start on the left bank
        OBS(env, i, LOC_COL + 0) = 1;

        // Set ENTITY_ID one-hot
        OBS(env, i, ENTITY_ID_COL + i) = 1;

    }  else if ((i < env->max_passengers * 2 + env->boats) && (i >= env->max_passengers * 2)) {
        OBS(env, i, TYPE_COL + 2) = 1; // boat

        // All entities start on the left bank
        OBS(env, i, LOC_COL + 0) = 1;

        // Set ENTITY_ID one-hot
        OBS(env, i, ENTITY_ID_COL + i) = 1;

        // no pair
    }
}


    env->boat_capacity =2;
    env->tick = 0;

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
    env->tick += 1;


    /* Termination shortcut  */
    #define FAIL(rew) do {env->rewards[0] = (rew); add_log(env); return; } while(0)

    if (boat_index < 0 || boat_index >= env->boats) FAIL(-1.0f);
    if (entity_index < 0 || entity_index >= env->passengers*2 + env->boats) FAIL(-1.0f);
    if (act_code < UNLOAD || act_code > MOVE) FAIL(-1.0f);

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
     

    }
    else if (act_code == LOAD) {

        if (entity_index >= env->passengers * 2) FAIL(-1.0f);  // cannot load/unload boats

        /* entity must share bank with boat and be boat‑less */
        int boat_row = boat_r + boat_index;
        int same_bank = 0;
        if (OBS(env, entity_index, LOC_COL)     && OBS(env, boat_row, LOC_COL))     same_bank = 1;
        if (OBS(env, entity_index, LOC_COL + 1) && OBS(env, boat_row, LOC_COL + 1)) same_bank = 1;
        if (!same_bank) FAIL(-1.0f);
       

        // add entity must not be in a boat condition

        if (
             in_a_boat(env, entity_index)
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

        for (int b = 0; b < env->boats; ++b) {
             OBS(env, entity_index, BOAT_LOC_COL + b) = 0;
            } // clear all boat locations for entity just in case

        OBS(env, entity_index, LOC_COL) = 0;
        OBS(env, entity_index, LOC_COL + 1) = 0;
        OBS(env, entity_index, BOAT_LOC_COL + boat_index) = 1;

        
        //env->rewards[0] = 0.2f; // loading gives small reward
       
    }


    else if (act_code == UNLOAD) {

        if (entity_index >= env->passengers * 2) FAIL(-1.0f);  // cannot load/unload boats

        /* entity must currently be in that boat */
        if (
            OBS(env, entity_index, BOAT_LOC_COL + boat_index) != 1
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
            

            //env->rewards[0] = 0.2f; // unloading gives small reward but only if on right bank
        } else {
            FAIL(-1.0f);
        }
        OBS(env, entity_index,  BOAT_LOC_COL + boat_index) = 0;
        
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
        //printf("Passenger-Agent conflict detected!\n");
       //fflush(stdout);
        return;
    }

    /* success if every passenger is on the right bank */
    int success = 1;
    for (int i = 0; i < env->passengers * 2; i ++) {   // all passenger rows and all agent rows
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
        //printf("All passengers on right bank! Success!\n");
        //fflush(stdout);
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


/* ------------------------------------- */
/*RENDERING*/
/* ------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"



/* --------------------------------------------------------------------------
 *  Raylib compatibility: rounded rect lines with thickness.
 * -------------------------------------------------------------------------- */
static inline void DrawRoundedRectLinesThick(Rectangle rec,
                                             float roundness,
                                             int segments,
                                             float lineThick,
                                             Color color)
{
#if defined(RAYLIB_VERSION_MAJOR) && (RAYLIB_VERSION_MAJOR >= 4)
    DrawRectangleRoundedLinesEx(rec, roundness, segments, lineThick, color);
#else
    (void)lineThick;
    DrawRectangleRoundedLines(rec, roundness, segments, color);
#endif
}

/* --------------------------------------------------------------------------
 *  Pair color palette (deterministic HSV ramp) -- FIXED ColorFromHSV usage.
 * -------------------------------------------------------------------------- */
static inline Color PairColor(int pairIdx, int totalPairs)
{
    if (totalPairs <= 0) return RAYWHITE;
    float h = fmodf((float)pairIdx * 360.0f / (float)totalPairs, 360.0f);
    return ColorFromHSV(h, 0.65f, 0.95f);  /* bright-ish */
}

/* --------------------------------------------------------------------------
 *  Layout constants (fractions of screen size)
 * -------------------------------------------------------------------------- */
#define RC_MARGIN_F         0.03f   /* outer margin % of width */
#define RC_BANK_W_F         0.20f   /* each bank width % of width */
#define RC_TOP_HUD_PAD      8       /* px from top for HUD */
#define RC_ICON_MIN         8       /* px min icon radius/half-size */
#define RC_ICON_MAX         28      /* px max icon radius/half-size */
#define RC_LABEL_PAD        4       /* px label offset from icon */

/* --------------------------------------------------------------------------
 *  Compute icon size based on available vertical space / entities.
 * -------------------------------------------------------------------------- */
static inline int ComputeIconSize(int availH, int totalEntities)
{
    if (totalEntities <= 0) return RC_ICON_MIN;
    int per = availH / totalEntities;
    if (per < RC_ICON_MIN) per = RC_ICON_MIN;
    if (per > RC_ICON_MAX) per = RC_ICON_MAX;
    return per;
}

/* --------------------------------------------------------------------------
 *  HUD drawing (FIX: no overlap)
 *
 *  We previously drew overlapping lines because we hard-coded Y and didn't
 *  advance between lines. Now we compute a font size from screen height, then
 *  measure text vertically and stack with padding.
 * -------------------------------------------------------------------------- */
static void DrawRC_HUD(RiverCrossing *env, int screenW, int screenH)
{
    (void)screenW; /* unused currently */
    const Font font = GetFontDefault();

    /* auto font size: small screens -> smaller font */
    int fontSize = screenH / 36;            /* ~13 @480h, scales up */
    if (fontSize < 10) fontSize = 10;
    if (fontSize > 28) fontSize = 28;

    float spacing = 0;                      /* default glyph spacing */

    char line1[64];
    char line2[64];
    snprintf(line1, sizeof(line1), "Tick: %d", env->tick);
    snprintf(line2, sizeof(line2), "Reward: %+.2f", env->rewards ? env->rewards[0] : 0.0f);

    Vector2 sz1 = MeasureTextEx(font, line1, (float)fontSize, spacing);
    Vector2 sz2 = MeasureTextEx(font, line2, (float)fontSize, spacing);

    float x = RC_TOP_HUD_PAD;
    float y = RC_TOP_HUD_PAD;

    /* background panel sized to max width */
    float pad = 4.0f;
    float w = (sz1.x > sz2.x ? sz1.x : sz2.x) + pad*2.0f;
    float h = sz1.y + sz2.y + pad*3.0f;   /* 1 pad above, 1 between, 1 below */
    DrawRectangleRounded((Rectangle){x-2, y-2, w+4, h+4}, 0.25f, 4, Fade(BLACK,0.5f));

    Vector2 pos1 = {x+pad, y+pad};
    DrawTextEx(font, line1, pos1, (float)fontSize, spacing, RAYWHITE);

    Vector2 pos2 = {x+pad, y+pad + sz1.y + pad};
    DrawTextEx(font, line2, pos2, (float)fontSize, spacing, RAYWHITE);
}

/* --------------------------------------------------------------------------
 *  Legend (toggle with L key) -- optional, drawn bottom-left.
 * -------------------------------------------------------------------------- */
static bool rc_show_legend = false;
static void DrawRC_Legend(RiverCrossing *env, int screenW, int screenH)
{
    if (!rc_show_legend) return;
    (void)env; /* not yet using env in legend */

    const Font font = GetFontDefault();
    int fontSize = screenH / 40; if (fontSize < 9) fontSize = 9; if (fontSize > 20) fontSize = 20;
    float spacing = 0;

    const char *lines[] = {
        "Legend:",
        "Filled circle = Passenger",
        "Square outline = Agent",
        "Brown hull = Boat",
        "Colors show pairs",
        NULL
    };

    float pad = 4.0f;
    float maxW = 0.0f; float totalH = pad;
    for (int i=0; lines[i]; ++i) {
        Vector2 sz = MeasureTextEx(font, lines[i], (float)fontSize, spacing);
        if (sz.x > maxW) maxW = sz.x;
        totalH += sz.y + pad;
    }
    float x = RC_TOP_HUD_PAD;
    float y = screenH - totalH - RC_TOP_HUD_PAD;
    DrawRectangleRounded((Rectangle){x-2,y-2,maxW+pad*2+4,totalH+4},0.25f,4,Fade(BLACK,0.5f));

    float cy = y + pad;
    for (int i=0; lines[i]; ++i) {
        DrawTextEx(font, lines[i], (Vector2){x+pad, cy}, (float)fontSize, spacing, RAYWHITE);
        Vector2 sz = MeasureTextEx(font, lines[i], (float)fontSize, spacing);
        cy += sz.y + pad;
    }
}

/* --------------------------------------------------------------------------
 *  Draw a boat hull + its occupants.
 * -------------------------------------------------------------------------- */
static void DrawRC_Boat(RiverCrossing *env, int boatIdx, Rectangle rect, int iconSz,
                        Color hullColor, Color outlineColor, Color labelColor,
                        int screenH)
{
    (void)screenH; /* reserved for future wave bobbing */

    /* Hull */
    DrawRectangleRec(rect, hullColor);
    DrawRoundedRectLinesThick(rect, 0.25f, 8, 2.0f, outlineColor);

    /* Occupants: gather entity indices */
    int rows_pa = env->passengers * 2;
    int totalSlots = env->boat_capacity; if (totalSlots < 1) totalSlots = 1; if (totalSlots > 8) totalSlots = 8;

    int occCount = 0;
    int occIdx[32];
    for (int e=0; e<rows_pa; ++e) {
        if (OBS(env, e, BOAT_LOC_COL + boatIdx) == 1) {
            if (occCount < (int)(sizeof(occIdx)/sizeof(occIdx[0]))) occIdx[occCount++] = e;
        }
    }

    /* layout occupant slots horizontally across rect */
    float pad = 2.0f;
    float slotW = rect.width / (float)totalSlots;
    float cxBase = rect.x + slotW * 0.5f;
    float cy = rect.y + rect.height * 0.5f;

    for (int s=0; s<occCount && s<totalSlots; ++s) {
        int ent = occIdx[s];
        int pair = ent/2; /* pair index */
        Color col = PairColor(pair, env->passengers);
        float cx = cxBase + s * slotW;
        if (is_passenger(ent)) {
            DrawCircle((int)cx, (int)cy, (float)iconSz*0.6f, col);
        } else {
            int hs = (int)(iconSz*0.6f);
            Rectangle r2 = {cx-hs, cy-hs, hs*2, hs*2};
            DrawRectangleLinesEx(r2, 2.0f, col);
        }
    }

    /* label */
    const Font font = GetFontDefault();
    int fontSize = rect.height * 0.45f; if (fontSize < 8) fontSize = 8; if (fontSize > 20) fontSize = 20;
    char lbl[16]; snprintf(lbl, sizeof(lbl), "B%d", boatIdx);
    Vector2 sz = MeasureTextEx(font, lbl, (float)fontSize, 0);
    DrawTextEx(font, lbl, (Vector2){rect.x + rect.width/2 - sz.x/2, rect.y - sz.y - 2}, (float)fontSize, 0, labelColor);
}

/* --------------------------------------------------------------------------
 *  Draw entity that is currently on a bank (not in boat)
 * -------------------------------------------------------------------------- */
static void DrawRC_EntityOnBank(RiverCrossing *env, int ent, int pairIdx, bool leftBank,
                                float x, float y, int iconSz)
{
    Color col = PairColor(pairIdx, env->passengers);
    if (ent >= env->passengers*2) {
        /* boats handled elsewhere */
        return;
    }

    if (is_passenger(ent)) {
        DrawCircle((int)x, (int)y, (float)iconSz, col);
    } else {
        Rectangle r = {x-iconSz, y-iconSz, (float)(iconSz*2), (float)(iconSz*2)};
        DrawRectangleLinesEx(r, 2.0f, col);
    }

    /* label */
    const Font font = GetFontDefault();
    int fontSize = iconSz; if (fontSize < 8) fontSize = 8; if (fontSize > 18) fontSize = 18;
    char lbl[16]; snprintf(lbl, sizeof(lbl), "%c%d", is_passenger(ent)?'P':'A', pairIdx);
    Vector2 sz = MeasureTextEx(font, lbl, (float)fontSize, 0);
    DrawTextEx(font, lbl, (Vector2){x - sz.x/2, y + iconSz + RC_LABEL_PAD}, (float)fontSize, 0, RAYWHITE);
    (void)leftBank; /* currently unused but reserved for direction-specific tweaks */
}

/* --------------------------------------------------------------------------
 *  Main render
 * -------------------------------------------------------------------------- */
void c_render(RiverCrossing *env)
{
    /* 1. Window init / teardown (lazy) ----------------------------------- */
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(640, 480, "RiverCrossing (PufferLib)");
        SetTargetFPS(60);   /* faster; will still be lightweight */
    }
    if (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE)) {
        CloseWindow();
        return;
    }

    /* Toggle legend */
    if (IsKeyPressed(KEY_L)) rc_show_legend = !rc_show_legend;

    /* 2. Layout ----------------------------------------------------------- */
    int screenW = GetScreenWidth();
    int screenH = GetScreenHeight();

    float margin = screenW * RC_MARGIN_F;
    float bankW  = screenW * RC_BANK_W_F;
    float riverX = margin + bankW;
    float riverW = screenW - margin*2 - bankW*2;
    float banksY = screenH * 0.15f;  /* top of playable strip */
    float banksH = screenH * 0.70f;  /* height of playable strip */

    Rectangle leftBank  = { margin,            banksY, bankW, banksH };
    Rectangle rightBank = { margin + bankW + riverW, banksY, bankW, banksH };
    Rectangle riverRect = { riverX, banksY, riverW, banksH };

    /* icon size = based on vertical space on bank / rows */
    int totalEntityRows = env->passengers * 2;  /* we only stack passengers+agents; boats drawn separately */
    int iconSz = ComputeIconSize((int)banksH, totalEntityRows);

    /* vertical spacing for stacking on bank */
    float stackGap = (banksH - iconSz*2) / (float)(totalEntityRows > 1 ? (totalEntityRows-1) : 1);
    if (stackGap < iconSz*0.5f) stackGap = iconSz*0.5f;

    /* Precompute Y for each row */
    float *rowY = (float*)alloca(sizeof(float) * (size_t)totalEntityRows);
    float cy = banksY + iconSz;
    for (int r=0; r<totalEntityRows; ++r) {
        rowY[r] = cy;
        cy += stackGap;
        if (cy > banksY + banksH - iconSz) cy = banksY + banksH - iconSz; /* clamp */
    }

    /* 3. Begin drawing ---------------------------------------------------- */
    BeginDrawing();
    ClearBackground((Color){24,24,24,255});

    /* River */
    DrawRectangleGradientV((int)riverRect.x, (int)riverRect.y, (int)riverRect.width, (int)riverRect.height,
                           (Color){0, 102, 204, 255}, (Color){0, 64, 160, 255});

    /* Banks */
    DrawRectangleRec(leftBank,  (Color){34,139,34,255});  /* green */
    DrawRectangleRec(rightBank, (Color){34,139,34,255});
    DrawRectangleLinesEx(leftBank,  2, Fade(BLACK,0.5f));
    DrawRectangleLinesEx(rightBank, 2, Fade(BLACK,0.5f));

    /* Bank labels */
    const Font font = GetFontDefault();
    int bankLblSize = screenH/30; if (bankLblSize < 12) bankLblSize = 12; if (bankLblSize>32) bankLblSize=32;
    Vector2 lsz = MeasureTextEx(font, "Left Bank", (float)bankLblSize, 0);
    Vector2 rsz = MeasureTextEx(font, "Right Bank", (float)bankLblSize, 0);
    DrawTextEx(font, "Left Bank",  (Vector2){leftBank.x + leftBank.width/2 - lsz.x/2, leftBank.y - lsz.y - 4},  (float)bankLblSize,0,RAYWHITE);
    DrawTextEx(font, "Right Bank", (Vector2){rightBank.x + rightBank.width/2 - rsz.x/2, rightBank.y - rsz.y - 4}, (float)bankLblSize,0,RAYWHITE);

    /* 4. Draw boats ------------------------------------------------------- */
    float boatH = iconSz * 2.2f; /* boat height scales with icon */
    float boatW = riverW * 0.18f; /* fraction of river width */
    if (boatW < iconSz*3) boatW = iconSz*3;
    if (boatW > riverW*0.4f) boatW = riverW*0.4f;

    Color hullColor   = (Color){139,69,19,255};  /* saddle brown */
    Color hullOutline = Fade(BLACK,0.8f);

    int boatBaseRow = env->max_passengers * 2;  /* where boat rows live in obs */

    for (int b=0; b<env->boats; ++b) {
        bool onLeft = OBS(env, boatBaseRow + b, LOC_COL) == 1;
        bool onRight= OBS(env, boatBaseRow + b, LOC_COL+1) == 1;

        float bx;
        if (onLeft)      bx = leftBank.x + leftBank.width + (riverW*0.05f);
        else if (onRight)bx = rightBank.x - (riverW*0.05f) - boatW;
        else             bx = riverX + riverW/2 - boatW/2;  /* mid-river fallback */

        float by = banksY + banksH*0.15f + b * (boatH + iconSz*0.5f);
        if (by + boatH > banksY + banksH) by = banksY + banksH - boatH;

        Rectangle br = {bx, by, boatW, boatH};
        DrawRC_Boat(env, b, br, iconSz, hullColor, hullOutline, RAYWHITE, screenH);
    }

    /* 5. Draw passengers & agents on banks (not in boats) ----------------- */
    for (int e=0; e<env->passengers*2; ++e) {
        if (in_a_boat(env, e)) continue;  /* drawn in boat */
        bool left  = OBS(env, e, LOC_COL)   == 1;
        bool right = OBS(env, e, LOC_COL+1) == 1;
        if (!left && !right) continue; /* something off-map? skip */

        float x = left ? (leftBank.x + leftBank.width*0.5f)
                       : (rightBank.x + rightBank.width*0.5f);
        float y = rowY[e];
        DrawRC_EntityOnBank(env, e, e/2, left, x, y, iconSz/2); /* /2 so icons fit nicely */
    }

    /* HUD (stacked lines; no overlap) */
    DrawRC_HUD(env, screenW, screenH);

    /* Legend if toggled */
    DrawRC_Legend(env, screenW, screenH);

    EndDrawing();
}

/* --------------------------------------------------------------------------
 *  Close (single authoritative definition)
 * -------------------------------------------------------------------------- */
void c_close(RiverCrossing *env)
{
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

