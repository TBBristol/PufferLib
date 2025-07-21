#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
#include <time.h>

/* CONSTS */

#define EMPTY_SLOT -1
#define STACK(env, s, h) (env->stacks[(s) * (env->max_height) + (h)])
#define OBS(env, s, f) (env->observations[(s) * 6 + f])

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
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int tick;
    // params
    int num_containers;
    int num_stacks;
    int max_height;
    int max_ep_steps;
    int seed;
    int unsorted;
    int next_container;
    int *container_leaving_priorities; // ptr to array of priorities for each container
    int *stacks; // ptr to array of stacks, each stack is an array of ints of size max_height
}ContainerStacking;

void add_log(ContainerStacking* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}



/* -----------------------------------------------------------
 * Helpers
 * -----------------------------------------------------------*/

 void sample_unique_integers(int *out, int n) {
    // Fill with 0..n-1
    for (int i = 0; i < n; i++) {
        out[i] = i;
    }

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }
}


 void initialise_stacks(ContainerStacking *env) {
    if (env->stacks != NULL) {
        free(env->stacks);
    }
    env->stacks = malloc(sizeof(int) * env->num_stacks * env->max_height);

    for (int i = 0; i < env->num_stacks * env->max_height; i++) {
        env->stacks[i] = EMPTY_SLOT;
    }
 }

 void generate_container_priorities(ContainerStacking *env) {

    /*Generate container leaving priorities this is a list
    where element 0 is the priority of the first container
    to arrive. Lower values indicate higher priority.*/

    if (env->container_leaving_priorities != NULL) {
        free(env->container_leaving_priorities);
    }
    env->container_leaving_priorities = malloc(sizeof(int) * env->num_containers);

    sample_unique_integers(env->container_leaving_priorities, env->num_containers);
    }
 


bool free_space(ContainerStacking *env,int stack) {

  //assume any free space means top slot is EMPTY

  return STACK(env, stack, env->max_height -1) == EMPTY_SLOT;
}

int find_next_height(ContainerStacking *env,int stack) {
    for (int h=0; h < env->max_height; h++){
        if (STACK(env, stack, h) == EMPTY_SLOT) {
            return h;
        }
    }
    return -1;
}

int find_height(ContainerStacking *env,int stack) {
    for (int h=0; h < env->max_height; h++){
        if (STACK(env, stack, h) == EMPTY_SLOT) {
            return h;
        }
    }
    return env->max_height;
    }

int count_unsorted_in_stack(ContainerStacking *env, int stack)
{
    int count = 0;
    for (int h = 1; h < env->max_height; h++) {
        if (STACK(env, stack, h) == EMPTY_SLOT)      // ⬅ break here
            break;
        if (STACK(env, stack, h) < STACK(env, stack, h - 1))
            count++;
    }
    return count;
}

int total_unsorted(ContainerStacking *env) {
    int count = 0;
    for (int s=0; s < env->num_stacks; s++){
        count += count_unsorted_in_stack(env, s);
    }
    return count;
}

int lowest_left(ContainerStacking *env) {
    int lowest = env->num_containers; // should be one higher than hiest possible generated

    //choice here we still need an obs if none left so if theres none left make it a really low priority
    if (env->next_container >= env-> num_containers){
        return 0;
    }
    for (int i = env->next_container; i < env->num_containers; i++) {
        if (env->container_leaving_priorities[i] < lowest) {
            lowest = env->container_leaving_priorities[i];
        }
    }
    return lowest;
}

int num_priority_less_than(ContainerStacking *env, int top){
    int count = 0;
    for (int i = env->next_container; i < env->num_containers; i++) {
        if (env->container_leaving_priorities[i] < top) {
            count +=1;
        }
    }
    return count;
}

int next_priority(ContainerStacking *env)
//choice here we still need an obs so if theres none left make it a really low priority
{
    if (env->next_container < env-> num_containers) {
        return env->container_leaving_priorities[env->next_container];
    }
    return env->num_containers;
}

bool stack_empty(ContainerStacking *env,int stack){
    if (STACK(env, stack, 0)== EMPTY_SLOT){
        return 1;
    }
    return 0;
}

int top_of_stack_priority(ContainerStacking *env,int stack){
    if (stack_empty(env,stack)){
        return 0; //effectively lowest priority so we want to stack here
    }
    return STACK(env, stack, find_next_height(env, stack) -1);
}

float* generate_obs(ContainerStacking *env){
    
    int lowest_remaining = lowest_left(env);

    //only accurate after next container has been iterated
    int nxt_container_priority = next_priority(env);

    for (int s=0; s< env->num_stacks; s++) {

        float height_percent = (float) find_height(env,s)/ (float) env->max_height;

        int top_prior = top_of_stack_priority(env,s);

        int num_less_than_top = num_priority_less_than(env, top_of_stack_priority(env,s));

        int stack_unsorted = count_unsorted_in_stack(env, s);

        OBS(env, s, 0) = height_percent;
        OBS(env, s, 1) = nxt_container_priority;
        OBS(env, s, 2) = top_prior;
        OBS(env, s, 3) = lowest_remaining;
        OBS(env, s, 4) = num_less_than_top;
        OBS(env, s, 5) = stack_unsorted;
    }
    return env->observations;
}


// Required function RESET

void c_reset(ContainerStacking *env) {

    //srand(((uintptr_t)env) ^ time(NULL)); //different seed per env and reset
  
    initialise_stacks(env);

    generate_container_priorities(env); //array each element int priority lowest num highest priority 

    //clear obs
    memset(env->observations, 0.0f, sizeof(float) * env->num_stacks * 6);

    env->tick = 0;
    env->unsorted = 0;
    env-> next_container = 0;

}   
    

// Required Function Step

/* OBS shape = (n_stacks, 6).
          (1) height%
          (2) next container priority
          (3) top container priority
          (4) lowest remaining priority
          (5) # of remaining containers with priority < top
          (6) # of unsorted in this stack*/

void c_step(ContainerStacking *env) {

    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;
    env->tick += 1;

    //check max steps exceeded

    if (env->tick >= env->max_ep_steps) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0f;
    add_log(env);
    c_reset(env);
    return;
}

    // Check if last container has been placed
    if (env->next_container >= env->num_containers) {
        env->terminals[0] = 1;
        env-> rewards[0] = 0; 
        add_log(env);
        c_reset(env);
        return;
    }

    int stack = env->actions[0]; // action is the stack to place the container in
    int old_unsorted = env-> unsorted;
    int container_priority = env->container_leaving_priorities[env->next_container];


    if (stack < 0 || stack >= env->num_stacks)  {
        env->terminals[0] = 1;
        env-> rewards[0] = -10.0f; 
        add_log(env);
        c_reset(env);
        return;
    }

    // Check if stack is valid
    if (!free_space(env, stack)){
        env->terminals[0] = 1;
        env-> rewards[0] = -10.0f; 
        add_log(env);
        c_reset(env);
        return;
    }

    //Place container
    int h = find_next_height(env, stack);
    if (h == -1) {
        fprintf(stderr, "Error no free slot but stack checks as valid");
        exit(1);  
    }
    STACK(env, stack, h) = container_priority;
    env->next_container += 1;


    //new unsorted
    env->unsorted = total_unsorted(env);
    
    //set new OBS
    generate_obs(env);

    // Set reward 
    env-> rewards[0] = (float) (old_unsorted - env->unsorted); // reward is the change in unsorted containers
    //printf("old unsorted %d, new unsorted %d\n", old_unsorted, env->unsorted);
    //fflush(stdout);

 
    /* no termination; step continues */
    return;
}


/* Tweak these if you want a different look */
#define CELL_W          100     // width of each table cell
#define CELL_H           20     // height of each table cell
#define OFFSET_X         20     // left margin
#define OFFSET_Y         20     // top margin

#define HEADER_COLOR     DARKGRAY
#define CELL_COLOR       BLACK
#define BG_COLOR         RAYWHITE
#define STACK_COL_X   (OFFSET_X + 6 * CELL_W + 20)   // x-pos of new column


static const char *OBS_HEADERS[6] = {
    "height%", "next", "top", "lowest", "<top", "unsrt"
};


// Required function. Should handle creating the client on first call
void c_render_old(ContainerStacking* env) {
    const int screenW = 800, screenH = 600;
    if (!IsWindowReady()) {
        InitWindow(screenW, screenH, "Container-Stacking visualiser");
        SetTargetFPS(2);
    }

     // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
 
    BeginDrawing();
    ClearBackground(BG_COLOR);

    /* ---------- Draw column headers ---------- */
    for (int f = 0; f < 6; f++) {
        DrawText(OBS_HEADERS[f],
                 OFFSET_X + f * CELL_W,
                 OFFSET_Y,
                 FONT_SIZE,
                 HEADER_COLOR);
    }
    /* extra header for stack contents */
    DrawText("stack", STACK_COL_X, OFFSET_Y, FONT_SIZE, HEADER_COLOR);

    /* ---------- Draw each observation row ---------- */
    for (int s = 0; s < env->num_stacks; s++) {
        /* Row label “S0”, “S1”, … */
        char rowLabel[8];
        snprintf(rowLabel, sizeof rowLabel, "S%d", s);
        DrawText(rowLabel,
                 OFFSET_X - 40,
                 OFFSET_Y + (s + 1) * CELL_H,
                 FONT_SIZE,
                 CELL_COLOR);


        /* The six float features */
        for (int f = 0; f < 6; f++) {
            char buf[16];
            snprintf(buf, sizeof buf, "%.2f", OBS(env, s, f));
            DrawText(buf,
                     OFFSET_X + f * CELL_W,
                     OFFSET_Y + (s + 1) * CELL_H,
                     FONT_SIZE,
                     CELL_COLOR);
        }
        /* ---- draw stack contents ---- */
        char cont[64] = "";
        for (int h = 0; h < env->max_height; h++) {
            int v = STACK(env, s, h);
            if (v == EMPTY_SLOT) break;
            char tmp[8];
            snprintf(tmp, sizeof tmp, "%d ", v);
            strncat(cont, tmp, sizeof cont - strlen(cont) - 1);
        }
        DrawText(cont, STACK_COL_X,
                 OFFSET_Y + (s + 1) * CELL_H,
                 FONT_SIZE,
                 CELL_COLOR);
    }    
    /* ---------- Remaining-to-stack list ---------- */
   int listY = OFFSET_Y + (env->num_stacks + 2) * CELL_H;
    DrawText("Remaining to stack:", OFFSET_X, listY, FONT_SIZE, HEADER_COLOR);

   listY += CELL_H;
   int listX = OFFSET_X;
   for (int i = env->next_container; i < env->num_containers; i++) {
       char num[8];
       snprintf(num, sizeof num, "%d", env->container_leaving_priorities[i]);
       DrawText(num, listX, listY, FONT_SIZE, CELL_COLOR);
      listX += CELL_W / 2;
      /* wrap to next line if we hit the right edge */
      if (listX + CELL_W / 2 > screenW) {
          listX = OFFSET_X;
          listY += CELL_H;
       }
   }
    EndDrawing();
}

void c_close(ContainerStacking *env)
{
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#define CELLH 40 //basic sizes to work in
#define CELLW 40
#define OFFSET_X 20     // left margin
#define OFFSET_Y 20     // top margin
#define SKYBLUE    CLITERAL(Color){ 102, 191, 255, 255 }   // Sky Blue
#define WHITE      CLITERAL(Color){ 255, 255, 255, 255 }   // White
#define FONT_SIZE        20

void c_render(ContainerStacking* env) {


    const int screenW = 800, screenH = 600;


    if (!IsWindowReady()) {
        InitWindow(screenW, screenH, "PufferLib Stacking");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int remaining_y = screenH - OFFSET_Y - CELLH * 3;
    int remaining_x = OFFSET_X;

    for (int i = env->next_container; i < env->num_containers; i++){

        DrawRectangle(remaining_x, remaining_y,
        CELLW, CELLH, SKYBLUE);

        int textx = remaining_x +10;
        int texty = remaining_y -10;
        
        char num[8];
        snprintf(num, sizeof num, "%d", env->container_leaving_priorities[i]);
        DrawText(num, textx, texty, FONT_SIZE, WHITE);
        


        remaining_x += CELLW + 20;
        if (remaining_x >= screenW - OFFSET_X - CELL_W) {
            remaining_y += 20 + CELLH;
            remaining_x = OFFSET_X;
        }
        if (remaining_y < screenH - OFFSET_Y + CELLH) {
            break;
        }

    }









    EndDrawing();

}